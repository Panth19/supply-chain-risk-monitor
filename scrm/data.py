"""Reproducible synthetic supply chain dataset.

The generator produces four tables that mirror what a demand/capacity planning
system typically works with:

- ``suppliers``   master data with lead-time and reliability characteristics
- ``skus``        article master data (category, supplier, unit cost)
- ``demand``      weekly demand history per SKU, including planned promotions
- ``deliveries``  historical purchase orders with promised vs. actual lead time
- ``inventory``   current on-hand stock and open purchase orders

Demand has a trend, yearly seasonality, promotion uplifts and overdispersed
(negative-binomial-like) noise. Suppliers differ in mean lead time, lead-time
variability and delay probability, so supplier risk is a real signal in the data
and not a constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

N_WEEKS = 156  # three years of weekly history
START_DATE = "2023-01-02"  # a Monday
CATEGORIES = ["Electronics", "Packaging", "Spare Parts"]

# id, name, mean lead time (weeks), lead-time std, delay probability, mean delay (weeks)
SUPPLIER_PROFILES = [
    ("SUP-01", "Rhein Components", 2.0, 0.3, 0.05, 1.0),
    ("SUP-02", "Ruhr Logistics Parts", 3.0, 0.6, 0.12, 1.5),
    ("SUP-03", "Baltic Packaging", 4.0, 1.0, 0.25, 2.0),
    ("SUP-04", "Overseas Electronics Ltd", 6.0, 1.5, 0.35, 3.0),
]


@dataclass
class Dataset:
    suppliers: pd.DataFrame
    skus: pd.DataFrame
    demand: pd.DataFrame
    deliveries: pd.DataFrame
    inventory: pd.DataFrame
    open_orders: pd.DataFrame

    @property
    def weeks(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.demand["week"].unique()))


def _suppliers() -> pd.DataFrame:
    return pd.DataFrame(
        SUPPLIER_PROFILES,
        columns=[
            "supplier_id",
            "name",
            "lead_time_mean_w",
            "lead_time_std_w",
            "delay_prob",
            "delay_mean_w",
        ],
    )


def generate(n_skus: int = 24, n_weeks: int = N_WEEKS, seed: int = 42) -> Dataset:
    """Generate the full dataset deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    suppliers = _suppliers()
    weeks = pd.date_range(START_DATE, periods=n_weeks, freq="W-MON")

    # ---- SKU master data ------------------------------------------------
    sku_rows = []
    for i in range(n_skus):
        sup = suppliers.iloc[i % len(suppliers)]
        sku_rows.append(
            {
                "sku_id": f"SKU-{i + 1:03d}",
                "category": CATEGORIES[i % len(CATEGORIES)],
                "supplier_id": sup.supplier_id,
                "unit_cost_eur": round(float(rng.uniform(2, 120)), 2),
                "base_level": float(rng.lognormal(mean=4.2, sigma=0.6)),  # ~65 units/week median
                "trend_per_year": float(rng.normal(0.05, 0.10)),
                "season_amp": float(rng.uniform(0.05, 0.45)),
                "season_phase": float(rng.uniform(0, 2 * np.pi)),
                "dispersion": float(rng.uniform(8, 30)),  # higher = less noise
                "promo_uplift": float(rng.uniform(0.3, 0.9)),
            }
        )
    skus = pd.DataFrame(sku_rows)

    # ---- Weekly demand ----------------------------------------------------
    t = np.arange(n_weeks)
    demand_rows = []
    for s in skus.itertuples():
        trend = 1 + s.trend_per_year * t / 52
        season = 1 + s.season_amp * np.sin(2 * np.pi * t / 52 + s.season_phase)
        # planned promotions: ~6% of weeks, known in advance by sales
        promo = (rng.random(n_weeks) < 0.06).astype(int)
        mu = s.base_level * np.clip(trend, 0.3, None) * season * (1 + s.promo_uplift * promo)
        # Gamma-Poisson mixture -> overdispersed counts
        lam = rng.gamma(shape=s.dispersion, scale=mu / s.dispersion)
        qty = rng.poisson(lam)
        for w, q, p in zip(weeks, qty, promo):
            demand_rows.append({"week": w, "sku_id": s.sku_id, "qty": int(q), "promo": int(p)})
    demand = pd.DataFrame(demand_rows)

    # Future promotions for the next 26 weeks (needed for forecasting)
    future_weeks = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=26, freq="W-MON")
    future_promo = [
        {"week": w, "sku_id": s, "qty": np.nan, "promo": int(rng.random() < 0.06)}
        for s in skus.sku_id
        for w in future_weeks
    ]
    demand = pd.concat([demand, pd.DataFrame(future_promo)], ignore_index=True)

    # ---- Historical deliveries (purchase orders) -------------------------
    sup_idx = suppliers.set_index("supplier_id")
    delivery_rows = []
    po = 1
    for s in skus.itertuples():
        prof = sup_idx.loc[s.supplier_id]
        order_weeks = weeks[: n_weeks - 8 : 4]  # one PO roughly every four weeks
        for ow in order_weeks:
            promised = max(1, int(np.rint(prof.lead_time_mean_w)))
            actual = max(1.0, rng.normal(prof.lead_time_mean_w, prof.lead_time_std_w))
            if rng.random() < prof.delay_prob:
                actual += rng.exponential(prof.delay_mean_w)
            actual_w = int(np.rint(actual))
            delivery_rows.append(
                {
                    "po_id": f"PO-{po:05d}",
                    "sku_id": s.sku_id,
                    "supplier_id": s.supplier_id,
                    "order_week": ow,
                    "promised_lead_time_w": promised,
                    "actual_lead_time_w": actual_w,
                    "delay_w": actual_w - promised,
                }
            )
            po += 1
    deliveries = pd.DataFrame(delivery_rows)

    # ---- Current inventory and open orders --------------------------------
    hist = demand.dropna(subset=["qty"])
    recent = hist[hist.week > weeks[-1] - pd.Timedelta(weeks=8)].groupby("sku_id").qty.mean()
    inv_rows, oo_rows = [], []
    for s in skus.itertuples():
        weekly = float(recent[s.sku_id])
        prof = sup_idx.loc[s.supplier_id]
        # weeks of cover varies a lot so that some SKUs are genuinely at risk
        cover = float(rng.uniform(0.5, 7.0))
        inv_rows.append({"sku_id": s.sku_id, "on_hand": int(weekly * cover)})
        # 0-2 open purchase orders with promised arrival weeks
        for _ in range(int(rng.integers(0, 3))):
            due_in = int(rng.integers(1, int(prof.lead_time_mean_w) + 2))
            oo_rows.append(
                {
                    "po_id": f"PO-{po:05d}",
                    "sku_id": s.sku_id,
                    "supplier_id": s.supplier_id,
                    "qty": int(weekly * rng.uniform(2, 5)),
                    "promised_in_w": due_in,
                }
            )
            po += 1
    inventory = pd.DataFrame(inv_rows)
    open_orders = pd.DataFrame(
        oo_rows, columns=["po_id", "sku_id", "supplier_id", "qty", "promised_in_w"]
    )

    skus_public = skus[["sku_id", "category", "supplier_id", "unit_cost_eur"]].copy()
    return Dataset(suppliers, skus_public, demand, deliveries, inventory, open_orders)


def save(ds: Dataset, out_dir: str) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    for name in ["suppliers", "skus", "demand", "deliveries", "inventory", "open_orders"]:
        getattr(ds, name).to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)


if __name__ == "__main__":
    ds = generate()
    save(ds, "data")
    print(
        f"wrote data/: {ds.skus.shape[0]} SKUs, {ds.suppliers.shape[0]} suppliers, "
        f"{ds.demand.qty.notna().sum()} demand rows, {len(ds.deliveries)} deliveries"
    )
