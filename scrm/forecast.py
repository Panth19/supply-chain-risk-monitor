"""Weekly demand forecasting.

Models compared in the backtest:

- ``naive``            last observed week
- ``seasonal_naive``   same week last year
- ``moving_avg_8``     mean of the last 8 weeks
- ``holt_winters``     damped additive trend + additive 52-week seasonality (statsmodels)
- ``gbm``              one global gradient-boosting model across all SKUs (scikit-learn),
                       direct multi-horizon, features scaled per SKU so that
                       high- and low-volume articles share one model

All models only see data up to the forecast origin; the GBM's training targets
also end at the origin, so the backtest has no look-ahead leakage. Planned
promotions are treated as known future inputs (sales plans them in advance).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

HORIZON = 12
MIN_HISTORY = 60  # weeks needed before the first GBM training origin


def to_wide(demand: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (qty, promo) as week x sku matrices. qty is NaN for future weeks."""
    qty = demand.pivot(index="week", columns="sku_id", values="qty").sort_index()
    promo = demand.pivot(index="week", columns="sku_id", values="promo").sort_index().fillna(0)
    return qty, promo


# ---------------------------------------------------------------- baselines
def naive(y: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(y[-1], h).astype(float)


def seasonal_naive(y: np.ndarray, h: int, season: int = 52) -> np.ndarray:
    return np.array([y[len(y) - season + i] for i in range(h)], dtype=float)


def moving_avg(y: np.ndarray, h: int, window: int = 8) -> np.ndarray:
    return np.repeat(y[-window:].mean(), h)


def holt_winters(y: np.ndarray, h: int) -> np.ndarray:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = ExponentialSmoothing(
            y.astype(float),
            trend="add",
            damped_trend=True,
            seasonal="add",
            seasonal_periods=52,
            initialization_method="estimated",
        ).fit()
    return np.clip(fit.forecast(h), 0, None)


# ---------------------------------------------------------------- GBM
FEATURES = [
    "h",
    "last1",
    "mean4",
    "mean8",
    "mean13",
    "std8",
    "lag52",
    "lag52_smooth",
    "woy_sin",
    "woy_cos",
    "promo",
    "promo_lag52",
    "cat",
]


def _features(
    y: np.ndarray, promo: np.ndarray, woy: np.ndarray, origin: int, h: int, cat: int
) -> tuple[dict, float]:
    """Features for predicting y[origin + h] using only y[: origin + 1]."""
    hist = y[: origin + 1]
    scale = hist[-26:].mean() + 1.0
    tgt = origin + h
    lag = tgt - 52
    return (
        {
            "h": h,
            "last1": hist[-1] / scale,
            "mean4": hist[-4:].mean() / scale,
            "mean8": hist[-8:].mean() / scale,
            "mean13": hist[-13:].mean() / scale,
            "std8": hist[-8:].std() / scale,
            "lag52": y[lag] / scale,
            "lag52_smooth": y[lag - 1 : lag + 2].mean() / scale,
            "woy_sin": np.sin(2 * np.pi * woy[tgt] / 52),
            "woy_cos": np.cos(2 * np.pi * woy[tgt] / 52),
            "promo": promo[tgt],
            "promo_lag52": promo[lag],
            "cat": cat,
        },
        scale,
    )


@dataclass
class GBMForecaster:
    horizon: int = HORIZON
    model: HistGradientBoostingRegressor = field(
        default_factory=lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, categorical_features=[FEATURES.index("cat")],
            random_state=0,
        )
    )

    def fit(self, qty: pd.DataFrame, promo: pd.DataFrame, cats: dict, last_obs: int):
        """Train on all (origin, h) pairs whose target index is <= last_obs."""
        woy = qty.index.isocalendar().week.to_numpy().astype(int)
        rows, targets = [], []
        for sku in qty.columns:
            y = qty[sku].to_numpy()
            p = promo[sku].to_numpy()
            for origin in range(MIN_HISTORY, last_obs):
                for h in range(1, self.horizon + 1):
                    if origin + h > last_obs:
                        break
                    f, scale = _features(y, p, woy, origin, h, cats[sku])
                    rows.append(f)
                    targets.append(y[origin + h] / scale)
        X = pd.DataFrame(rows)[FEATURES]
        self.model.fit(X, np.asarray(targets))
        return self

    def predict(
        self, qty: pd.DataFrame, promo: pd.DataFrame, cats: dict, origin: int
    ) -> pd.DataFrame:
        woy = qty.index.isocalendar().week.to_numpy().astype(int)
        out = {}
        for sku in qty.columns:
            y = qty[sku].to_numpy()
            p = promo[sku].to_numpy()
            rows, scales = [], []
            for h in range(1, self.horizon + 1):
                f, s = _features(y, p, woy, origin, h, cats[sku])
                rows.append(f)
                scales.append(s)
            pred = self.model.predict(pd.DataFrame(rows)[FEATURES]) * np.array(scales)
            out[sku] = np.clip(pred, 0, None)
        return pd.DataFrame(out, index=qty.index[origin + 1 : origin + 1 + self.horizon])


def category_codes(skus: pd.DataFrame) -> dict:
    codes = {c: i for i, c in enumerate(sorted(skus.category.unique()))}
    return {r.sku_id: codes[r.category] for r in skus.itertuples()}


# ---------------------------------------------------------------- metrics
def wape(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).sum() / actual.sum())


def mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.abs(actual - pred).mean())


def bias(actual: np.ndarray, pred: np.ndarray) -> float:
    return float((pred - actual).sum() / actual.sum())
