"""Stockout and supplier risk.

Stockout risk (per SKU) is estimated by Monte Carlo simulation over the
forecast horizon. Each simulation run draws

- weekly demand  = forecast x a log-normal error whose spread is taken from the
                   backtest's empirical p10/p90 forecast errors for that horizon
- PO arrivals    = promised arrival week + a delay sampled from that supplier's
                   own delivery history

and steps inventory week by week (lost sales, no backorders). The risk window
is the SKU's replenishment lead time (supplier mean lead time + its average
delay): a shortage inside that window cannot be fixed by ordering today, only
by expediting. The share of runs in which inventory runs out inside the window
is the stockout probability.

Supplier risk is a transparent heuristic score (0-100) built from the delivery
history: on-time rate, lead-time variability and average delay.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z90 = 1.2816  # standard-normal quantile for 90%


def _sigma_per_horizon(intervals: dict, horizon: int) -> np.ndarray:
    sig = []
    for h in range(1, horizon + 1):
        m = intervals[str(h)] if str(h) in intervals else intervals[h]
        sig.append((np.log(m["p90"]) - np.log(m["p10"])) / (2 * Z90))
    return np.array(sig)


def simulate_stockouts(
    forecast: pd.DataFrame,
    intervals: dict,
    inventory: pd.DataFrame,
    open_orders: pd.DataFrame,
    deliveries: pd.DataFrame,
    skus: pd.DataFrame,
    suppliers: pd.DataFrame,
    review_period: int = 4,
    service_quantile: float = 0.95,
    n_sims: int = 2000,
    seed: int = 7,
) -> pd.DataFrame:
    """Return one row per SKU with stockout probability and an order suggestion."""
    rng = np.random.default_rng(seed)
    H = forecast.shape[0]
    sigma = _sigma_per_horizon(intervals, H)
    on_hand = inventory.set_index("sku_id").on_hand
    sku_info = skus.set_index("sku_id")
    sup_lt = suppliers.set_index("supplier_id").lead_time_mean_w
    delays_by_sup = {s: g.delay_w.to_numpy() for s, g in deliveries.groupby("supplier_id")}

    rows = []
    for sku in forecast.columns:
        f = forecast[sku].to_numpy()
        sup = sku_info.loc[sku, "supplier_id"]
        delays = delays_by_sup[sup]

        # demand paths: (n_sims, H)
        noise = rng.lognormal(mean=-0.5 * sigma**2, sigma=sigma, size=(n_sims, H))
        demand = rng.poisson(f * noise)

        # PO arrivals with sampled delays
        arrivals = np.zeros((n_sims, H + 1))
        sku_orders = open_orders[open_orders.sku_id == sku]
        for po in sku_orders.itertuples():
            wk = np.clip(po.promised_in_w + rng.choice(delays, n_sims), 1, H)
            arrivals[np.arange(n_sims), wk] += po.qty

        # risk window = time until an order placed today would arrive
        lead = int(np.clip(round(sup_lt[sup] + max(delays.mean(), 0)), 1, H))
        risk_horizon = lead
        inv = np.full(n_sims, float(on_hand[sku]))
        stocked_out = np.zeros(n_sims, bool)
        first_out = np.full(n_sims, np.nan)
        shortfall = np.zeros(n_sims)
        for w in range(1, risk_horizon + 1):
            inv += arrivals[:, w]
            inv -= demand[:, w - 1]
            new_out = (inv < 0) & ~stocked_out
            first_out[new_out] = w
            stocked_out |= inv < 0
            shortfall += np.clip(-inv, 0, None)
            inv = np.clip(inv, 0, None)

        # order suggestion: cover lead time (+ typical delay) + review period
        cover_weeks = min(lead + review_period, H)
        need = np.quantile(demand[:, :cover_weeks].sum(axis=1), service_quantile)
        position = on_hand[sku] + sku_orders.qty.sum()
        order_qty = int(max(0, np.ceil(need - position)))

        p = float(stocked_out.mean())
        rows.append(
            {
                "sku_id": sku,
                "supplier_id": sup,
                "category": sku_info.loc[sku, "category"],
                "on_hand": int(on_hand[sku]),
                "open_order_qty": int(sku_orders.qty.sum()),
                "replenishment_lead_time_w": lead,
                "weeks_of_cover": round(float(on_hand[sku] / max(f[:4].mean(), 1e-9)), 1),
                "stockout_probability": round(p, 3),
                "median_weeks_to_stockout": (
                    None if np.isnan(first_out).all() else float(np.nanmedian(first_out))
                ),
                "expected_shortfall_units": round(float(shortfall.mean()), 1),
                "expected_shortfall_value_eur": round(
                    float(shortfall.mean() * sku_info.loc[sku, "unit_cost_eur"]), 2
                ),
                "suggested_order_qty": order_qty,
                "risk_level": "high" if p >= 0.5 else "medium" if p >= 0.2 else "low",
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["stockout_probability", "expected_shortfall_value_eur"], ascending=False
    ).reset_index(drop=True)


def supplier_risk(
    deliveries: pd.DataFrame, suppliers: pd.DataFrame, skus: pd.DataFrame
) -> pd.DataFrame:
    """Heuristic supplier risk score from delivery history (higher = riskier).

    score = 100 * (0.5 * late_rate + 0.3 * min(lead_time_cv / 0.5, 1)
                   + 0.2 * min(mean_delay_w / 2, 1))
    """
    rows = []
    for sup, g in deliveries.groupby("supplier_id"):
        late_rate = float((g.delay_w > 0).mean())
        cv = float(g.actual_lead_time_w.std() / g.actual_lead_time_w.mean())
        mean_delay = float(g.delay_w.clip(lower=0).mean())
        score = 100 * (0.5 * late_rate + 0.3 * min(cv / 0.5, 1) + 0.2 * min(mean_delay / 2, 1))
        rows.append(
            {
                "supplier_id": sup,
                "name": suppliers.set_index("supplier_id").loc[sup, "name"],
                "deliveries": len(g),
                "on_time_rate": round(1 - late_rate, 3),
                "mean_lead_time_w": round(float(g.actual_lead_time_w.mean()), 2),
                "lead_time_cv": round(cv, 3),
                "mean_delay_w": round(mean_delay, 2),
                "p95_delay_w": float(np.quantile(g.delay_w, 0.95)),
                "skus_supplied": int((skus.supplier_id == sup).sum()),
                "risk_score": round(score, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def summarize(stockouts: pd.DataFrame, suppliers: pd.DataFrame) -> str:
    """Plain-language summary of the current risk picture (rule-based)."""
    high = stockouts[stockouts.risk_level == "high"]
    med = stockouts[stockouts.risk_level == "medium"]
    lines = [
        (
            f"{len(high)} of {len(stockouts)} SKUs have a stockout probability of 50% or more "
            f"before a new order could arrive; {len(med)} more are at medium risk (20-50%)."
        )
    ]
    if len(high):
        total = high.expected_shortfall_value_eur.sum()
        top = high.head(3)
        names = ", ".join(f"{r.sku_id} ({r.stockout_probability:.0%})" for r in top.itertuples())
        lines.append(f"Most urgent: {names}. Expected shortfall value of high-risk SKUs: "
                     f"EUR {total:,.0f}.")
        by_sup = high.groupby("supplier_id").size().sort_values(ascending=False)
        lines.append(
            "High-risk SKUs by supplier: "
            + ", ".join(f"{s}: {n}" for s, n in by_sup.items())
            + "."
        )
    worst = suppliers.iloc[0]
    lines.append(
        f"Riskiest supplier: {worst['name']} ({worst.supplier_id}), on-time rate "
        f"{worst.on_time_rate:.0%}, average delay {worst.mean_delay_w:.1f} weeks."
    )
    return " ".join(lines)
