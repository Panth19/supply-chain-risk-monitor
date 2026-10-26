"""FastAPI service exposing forecasts and risk indicators.

Run:  uvicorn scrm.api:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .engine import PlanningState, build

STATE: dict[str, PlanningState] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    STATE["s"] = build()
    yield
    STATE.clear()


app = FastAPI(
    title="Supply Chain Risk Monitor",
    version=__version__,
    description="Demand forecasts, stockout probabilities and supplier risk scores.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"])


def state() -> PlanningState:
    return STATE["s"]


# ---------------------------------------------------------------- schemas
class Sku(BaseModel):
    sku_id: str
    category: str
    supplier_id: str
    unit_cost_eur: float


class Point(BaseModel):
    week: str
    qty: float
    p10: float | None = None
    p90: float | None = None


class ForecastResponse(BaseModel):
    sku_id: str
    history: list[Point]
    forecast: list[Point]


class StockoutRisk(BaseModel):
    sku_id: str
    supplier_id: str
    category: str
    on_hand: int
    open_order_qty: int
    replenishment_lead_time_w: int
    weeks_of_cover: float
    stockout_probability: float
    median_weeks_to_stockout: float | None
    expected_shortfall_units: float
    expected_shortfall_value_eur: float
    suggested_order_qty: int
    risk_level: Literal["high", "medium", "low"]


class SupplierRisk(BaseModel):
    supplier_id: str
    name: str
    deliveries: int
    on_time_rate: float
    mean_lead_time_w: float
    lead_time_cv: float
    mean_delay_w: float
    p95_delay_w: float
    skus_supplied: int
    risk_score: float


class Summary(BaseModel):
    as_of_week: str
    text: str
    high_risk_skus: int
    medium_risk_skus: int
    forecast_wape_backtest: float


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.get("/api/skus", response_model=list[Sku])
def list_skus():
    return state().ds.skus.to_dict("records")


@app.get("/api/skus/{sku_id}/forecast", response_model=ForecastResponse)
def sku_forecast(sku_id: str, history_weeks: int = Query(52, ge=4, le=156)):
    s = state()
    if sku_id not in s.forecast.columns:
        raise HTTPException(404, f"Unknown SKU {sku_id}")
    hist = s.qty[sku_id].tail(history_weeks)
    mult = s.backtest["gbm_interval_multipliers"]
    fc_points = [
        Point(
            week=str(w.date()),
            qty=round(float(v), 1),
            p10=round(float(v) * mult[str(i + 1)]["p10"], 1),
            p90=round(float(v) * mult[str(i + 1)]["p90"], 1),
        )
        for i, (w, v) in enumerate(s.forecast[sku_id].items())
    ]
    return ForecastResponse(
        sku_id=sku_id,
        history=[Point(week=str(w.date()), qty=float(v)) for w, v in hist.items()],
        forecast=fc_points,
    )


@app.get("/api/risk/stockouts", response_model=list[StockoutRisk])
def stockout_risk(level: Literal["high", "medium", "low"] | None = None):
    df = state().stockouts
    if level:
        df = df[df.risk_level == level]
    return df.to_dict("records")


@app.get("/api/risk/suppliers", response_model=list[SupplierRisk])
def suppliers_risk():
    return state().suppliers.to_dict("records")


@app.get("/api/summary", response_model=Summary)
def summary():
    s = state()
    return Summary(
        as_of_week=str(s.last_week.date()),
        text=s.summary,
        high_risk_skus=int((s.stockouts.risk_level == "high").sum()),
        medium_risk_skus=int((s.stockouts.risk_level == "medium").sum()),
        forecast_wape_backtest=s.backtest["models"]["gbm"]["wape"],
    )


@app.get("/api/backtest")
def backtest():
    return state().backtest


# Serve the built dashboard (frontend/dist) at / when it exists
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount("/", StaticFiles(directory=_DIST, html=True), name="dashboard")
