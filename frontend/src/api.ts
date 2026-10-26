// Typed client for the FastAPI backend. Types mirror the Pydantic models in scrm/api.py.

export type RiskLevel = "high" | "medium" | "low";

export interface Point {
  week: string;
  qty: number;
  p10: number | null;
  p90: number | null;
}

export interface ForecastResponse {
  sku_id: string;
  history: Point[];
  forecast: Point[];
}

export interface StockoutRisk {
  sku_id: string;
  supplier_id: string;
  category: string;
  on_hand: number;
  open_order_qty: number;
  replenishment_lead_time_w: number;
  weeks_of_cover: number;
  stockout_probability: number;
  median_weeks_to_stockout: number | null;
  expected_shortfall_units: number;
  expected_shortfall_value_eur: number;
  suggested_order_qty: number;
  risk_level: RiskLevel;
}

export interface SupplierRisk {
  supplier_id: string;
  name: string;
  deliveries: number;
  on_time_rate: number;
  mean_lead_time_w: number;
  lead_time_cv: number;
  mean_delay_w: number;
  p95_delay_w: number;
  skus_supplied: number;
  risk_score: number;
}

export interface Summary {
  as_of_week: string;
  text: string;
  high_risk_skus: number;
  medium_risk_skus: number;
  forecast_wape_backtest: number;
}

export interface Backtest {
  models: Record<string, { wape: number; mae: number; bias: number }>;
  best_baseline: string;
  gbm_wape_reduction_vs_best_baseline: number;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  summary: () => get<Summary>("/api/summary"),
  stockouts: (level?: RiskLevel) =>
    get<StockoutRisk[]>(`/api/risk/stockouts${level ? `?level=${level}` : ""}`),
  suppliers: () => get<SupplierRisk[]>("/api/risk/suppliers"),
  forecast: (sku: string) => get<ForecastResponse>(`/api/skus/${encodeURIComponent(sku)}/forecast`),
  backtest: () => get<Backtest>("/api/backtest"),
};
