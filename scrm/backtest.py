"""Rolling-origin backtest of all forecasting models.

Three forecast origins, each predicting the next 12 weeks, so the evaluation
covers the last 36 weeks of history (24 SKUs x 36 weeks = 864 forecast points
per model). Writes ``reports/backtest.json``.

Run:  python -m scrm.backtest
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd

from . import forecast as fc
from .data import generate

N_ORIGINS = 3


def run(ds=None, horizon: int = fc.HORIZON, n_origins: int = N_ORIGINS) -> dict:
    ds = ds or generate()
    qty, promo = fc.to_wide(ds.demand)
    n_obs = int(qty.notna().all(axis=1).sum())
    cats = fc.category_codes(ds.skus)
    origins = [n_obs - 1 - horizon * k for k in range(n_origins, 0, -1)]

    baselines = {
        "naive": fc.naive,
        "seasonal_naive": fc.seasonal_naive,
        "moving_avg_8": fc.moving_avg,
        "holt_winters": fc.holt_winters,
    }
    preds = {m: [] for m in [*baselines, "gbm"]}
    actuals, gbm_ratio = [], {h: [] for h in range(1, horizon + 1)}
    timings = {m: 0.0 for m in preds}

    for origin in origins:
        act = qty.iloc[origin + 1 : origin + 1 + horizon].to_numpy()
        actuals.append(act)
        for name, fn in baselines.items():
            t0 = time.perf_counter()
            p = np.column_stack(
                [fn(qty[s].to_numpy()[: origin + 1], horizon) for s in qty.columns]
            )
            timings[name] += time.perf_counter() - t0
            preds[name].append(p)
        t0 = time.perf_counter()
        gbm = fc.GBMForecaster(horizon).fit(qty, promo, cats, last_obs=origin)
        p = gbm.predict(qty, promo, cats, origin).to_numpy()
        timings["gbm"] += time.perf_counter() - t0
        preds["gbm"].append(p)
        for h in range(horizon):
            gbm_ratio[h + 1].extend((act[h] / np.maximum(p[h], 1.0)).tolist())

    A = np.concatenate(actuals)
    results = {}
    for name, ps in preds.items():
        P = np.concatenate(ps)
        results[name] = {
            "wape": round(fc.wape(A, P), 4),
            "mae": round(fc.mae(A, P), 2),
            "bias": round(fc.bias(A, P), 4),
            "seconds": round(timings[name], 2),
        }

    # Per-horizon WAPE for the GBM (does error grow with horizon?)
    P = np.concatenate(preds["gbm"])
    per_h = {}
    for h in range(horizon):
        rows = slice(h, None, horizon)
        per_h[h + 1] = round(fc.wape(A[rows], P[rows]), 4)

    # Empirical interval multipliers (actual / forecast), used by the API
    intervals = {
        h: {"p10": round(float(np.quantile(r, 0.10)), 3), "p90": round(float(np.quantile(r, 0.90)), 3)}
        for h, r in gbm_ratio.items()
    }
    lo = np.array([intervals[(i % horizon) + 1]["p10"] for i in range(P.shape[0])])[:, None]
    hi = np.array([intervals[(i % horizon) + 1]["p90"] for i in range(P.shape[0])])[:, None]
    coverage = float(np.mean((A >= lo * P) & (A <= hi * P)))

    best_baseline = min(baselines, key=lambda m: results[m]["wape"])
    improvement = 1 - results["gbm"]["wape"] / results[best_baseline]["wape"]
    return {
        "setup": {
            "skus": int(qty.shape[1]),
            "origins": [str(qty.index[o].date()) for o in origins],
            "horizon_weeks": horizon,
            "forecast_points_per_model": int(A.size),
        },
        "models": results,
        "best_baseline": best_baseline,
        "gbm_wape_reduction_vs_best_baseline": round(float(improvement), 4),
        "gbm_wape_by_horizon": per_h,
        "gbm_interval_multipliers": intervals,
        "gbm_interval_coverage_p10_p90_in_sample": round(coverage, 4),
    }


if __name__ == "__main__":
    res = run()
    os.makedirs("reports", exist_ok=True)
    with open("reports/backtest.json", "w") as f:
        json.dump(res, f, indent=2)
    df = pd.DataFrame(res["models"]).T.sort_values("wape")
    print(df.to_string())
    print(
        f"\nGBM WAPE reduction vs best baseline ({res['best_baseline']}): "
        f"{res['gbm_wape_reduction_vs_best_baseline']:.1%}"
    )
