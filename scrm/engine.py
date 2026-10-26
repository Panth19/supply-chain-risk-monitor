"""Builds the full planning state once: data, forecast model, risk tables."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import pandas as pd

from . import forecast as fc
from . import risk
from .backtest import run as run_backtest
from .data import Dataset, generate

REPORT_PATH = os.environ.get("SCRM_BACKTEST_REPORT", "reports/backtest.json")


@dataclass
class PlanningState:
    ds: Dataset
    qty: pd.DataFrame
    forecast: pd.DataFrame
    backtest: dict
    stockouts: pd.DataFrame
    suppliers: pd.DataFrame
    summary: str
    last_week: pd.Timestamp


def build(seed: int = 42) -> PlanningState:
    ds = generate(seed=seed)
    qty, promo = fc.to_wide(ds.demand)
    n_obs = int(qty.notna().all(axis=1).sum())
    origin = n_obs - 1
    cats = fc.category_codes(ds.skus)

    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            bt = json.load(f)
    else:
        bt = run_backtest(ds)

    model = fc.GBMForecaster().fit(qty, promo, cats, last_obs=origin)
    forecast = model.predict(qty, promo, cats, origin)

    stockouts = risk.simulate_stockouts(
        forecast,
        bt["gbm_interval_multipliers"],
        ds.inventory,
        ds.open_orders,
        ds.deliveries,
        ds.skus,
        ds.suppliers,
    )
    suppliers = risk.supplier_risk(ds.deliveries, ds.suppliers, ds.skus)
    return PlanningState(
        ds=ds,
        qty=qty.iloc[: origin + 1],
        forecast=forecast,
        backtest=bt,
        stockouts=stockouts,
        suppliers=suppliers,
        summary=risk.summarize(stockouts, suppliers),
        last_week=qty.index[origin],
    )
