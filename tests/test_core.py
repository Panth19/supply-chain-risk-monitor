import numpy as np
import pandas as pd
import pytest

from scrm import forecast as fc
from scrm import risk
from scrm.data import generate


@pytest.fixture(scope="module")
def ds():
    return generate(n_skus=6, seed=1)


def test_generator_is_deterministic():
    a, b = generate(n_skus=4, seed=3), generate(n_skus=4, seed=3)
    pd.testing.assert_frame_equal(a.demand, b.demand)
    pd.testing.assert_frame_equal(a.deliveries, b.deliveries)


def test_demand_is_non_negative_and_weekly(ds):
    hist = ds.demand.dropna(subset=["qty"])
    assert (hist.qty >= 0).all()
    weeks = ds.weeks
    assert (np.diff(weeks.values).astype("timedelta64[D]").astype(int) == 7).all()


def test_gbm_has_no_lookahead_leakage(ds):
    """Corrupting every value after the origin must not change the forecast."""
    qty, promo = fc.to_wide(ds.demand)
    cats = fc.category_codes(ds.skus)
    origin = 120
    model = fc.GBMForecaster(horizon=6).fit(qty, promo, cats, last_obs=origin)
    p1 = model.predict(qty, promo, cats, origin)

    corrupted = qty.copy()
    corrupted.iloc[origin + 1 :] = 1e6
    p2 = model.predict(corrupted, promo, cats, origin)
    pd.testing.assert_frame_equal(p1, p2)


def test_baselines_shapes():
    y = np.arange(120, dtype=float)
    assert fc.naive(y, 5).tolist() == [119.0] * 5
    assert fc.seasonal_naive(y, 3).tolist() == [68.0, 69.0, 70.0]
    assert fc.moving_avg(y, 2, window=4).tolist() == [117.5, 117.5]


def test_metrics():
    a, p = np.array([10.0, 10.0]), np.array([8.0, 14.0])
    assert fc.wape(a, p) == pytest.approx(0.3)
    assert fc.mae(a, p) == pytest.approx(3.0)
    assert fc.bias(a, p) == pytest.approx(0.1)


def _risk_inputs(ds, on_hand_value):
    forecast = pd.DataFrame(
        {s: np.full(12, 50.0) for s in ds.skus.sku_id},
        index=pd.date_range("2026-01-05", periods=12, freq="W-MON"),
    )
    intervals = {str(h): {"p10": 0.8, "p90": 1.25} for h in range(1, 13)}
    inv = pd.DataFrame({"sku_id": ds.skus.sku_id, "on_hand": on_hand_value})
    no_orders = ds.open_orders.iloc[0:0]
    return forecast, intervals, inv, no_orders


def test_stockout_extremes(ds):
    f, iv, _, oo = _risk_inputs(ds, 0)
    empty = risk.simulate_stockouts(f, iv, pd.DataFrame({"sku_id": ds.skus.sku_id, "on_hand": 0}),
                                    oo, ds.deliveries, ds.skus, ds.suppliers, n_sims=300)
    full = risk.simulate_stockouts(f, iv, pd.DataFrame({"sku_id": ds.skus.sku_id, "on_hand": 10**6}),
                                   oo, ds.deliveries, ds.skus, ds.suppliers, n_sims=300)
    assert (empty.stockout_probability == 1.0).all()
    assert (empty.risk_level == "high").all()
    assert (full.stockout_probability == 0.0).all()
    assert (full.suggested_order_qty == 0).all()


def test_open_orders_reduce_risk(ds):
    f, iv, _, oo = _risk_inputs(ds, 0)
    inv = pd.DataFrame({"sku_id": ds.skus.sku_id, "on_hand": 60})
    base = risk.simulate_stockouts(f, iv, inv, oo, ds.deliveries, ds.skus, ds.suppliers, n_sims=500)
    orders = pd.DataFrame(
        [{"po_id": f"X{i}", "sku_id": s, "supplier_id": sup, "qty": 2000, "promised_in_w": 1}
         for i, (s, sup) in enumerate(zip(ds.skus.sku_id, ds.skus.supplier_id))]
    )
    helped = risk.simulate_stockouts(f, iv, inv, orders, ds.deliveries, ds.skus, ds.suppliers,
                                     n_sims=500)
    m = base.merge(helped, on="sku_id", suffixes=("_base", "_po"))
    assert (m.stockout_probability_po <= m.stockout_probability_base).all()
    assert m.stockout_probability_po.mean() < m.stockout_probability_base.mean()


def test_supplier_risk_ranks_unreliable_supplier_higher():
    full = generate(seed=42)
    sr = risk.supplier_risk(full.deliveries, full.suppliers, full.skus).set_index("supplier_id")
    # SUP-04 is configured with the highest delay probability, SUP-01 the lowest
    assert sr.loc["SUP-04", "risk_score"] > sr.loc["SUP-01", "risk_score"]
    assert sr.risk_score.between(0, 100).all()
