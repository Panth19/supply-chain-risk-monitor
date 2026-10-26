import {
  BarController,
  BarElement,
  CategoryScale,
  Chart,
  Filler,
  Legend,
  LinearScale,
  LineController,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { api, type RiskLevel, type StockoutRisk, type SupplierRisk } from "./api";
import "./style.css";

Chart.register(
  BarController, BarElement, CategoryScale, Filler, Legend,
  LinearScale, LineController, LineElement, PointElement, Tooltip,
);

const $ = <T extends HTMLElement>(sel: string) => document.querySelector(sel) as T;
const css = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const pct = (x: number) => `${Math.round(x * 100)}%`;
const eur = (x: number) =>
  new Intl.NumberFormat("de-DE", { style: "currency", currency: "EUR", maximumFractionDigits: 0 }).format(x);
const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

let forecastChart: Chart | null = null;
let selectedSku: string | null = null;
let supplierNames: Record<string, string> = {};

// ------------------------------------------------------------------ KPIs
function renderKpis(high: number, medium: number, shortfall: number, wape: number, gain: number, best: string) {
  const items = [
    { label: "High-risk SKUs", value: String(high), tone: "high" },
    { label: "Medium-risk SKUs", value: String(medium), tone: "medium" },
    { label: "Expected shortfall (high risk)", value: eur(shortfall), tone: "" },
    {
      label: "Forecast error (WAPE)",
      value: pct(wape),
      sub: `${pct(gain)} lower than ${best.replace(/_/g, "-")}`,
      tone: "",
    },
  ];
  $("#kpis").innerHTML = items
    .map(
      (k) => `<div class="kpi ${k.tone}">
        <div class="kpi-label">${k.label}</div>
        <div class="kpi-value">${k.value}</div>
        ${k.sub ? `<div class="kpi-sub">${k.sub}</div>` : ""}
      </div>`,
    )
    .join("");
}

// ------------------------------------------------------------------ tables
function renderStockouts(rows: StockoutRisk[]) {
  const tbody = $<HTMLTableSectionElement>("#stockout-table tbody");
  tbody.innerHTML = rows
    .map(
      (r) => `<tr data-sku="${esc(r.sku_id)}" class="${r.sku_id === selectedSku ? "selected" : ""}" tabindex="0">
        <td><span class="dot ${r.risk_level}" aria-hidden="true"></span>${esc(r.sku_id)}
            <span class="muted small">${esc(r.category)}</span></td>
        <td title="${esc(supplierNames[r.supplier_id] ?? "")}">${esc(r.supplier_id)}</td>
        <td class="num">${r.weeks_of_cover.toFixed(1)}</td>
        <td class="num">${r.replenishment_lead_time_w} wk</td>
        <td class="num"><span class="pill ${r.risk_level}">${pct(r.stockout_probability)}</span></td>
        <td class="num">${eur(r.expected_shortfall_value_eur)}</td>
        <td class="num">${r.suggested_order_qty || "–"}</td>
      </tr>`,
    )
    .join("");
  tbody.querySelectorAll<HTMLTableRowElement>("tr").forEach((tr) => {
    const pick = () => selectSku(tr.dataset.sku!);
    tr.addEventListener("click", pick);
    tr.addEventListener("keydown", (e) => e.key === "Enter" && pick());
  });
}

function renderSuppliers(rows: SupplierRisk[]) {
  $<HTMLTableSectionElement>("#supplier-table tbody").innerHTML = rows
    .map((s) => {
      const tone = s.risk_score >= 50 ? "high" : s.risk_score >= 25 ? "medium" : "low";
      return `<tr>
        <td>${esc(s.name)} <span class="muted small">${esc(s.supplier_id)} · ${s.skus_supplied} SKUs</span></td>
        <td class="num">${pct(s.on_time_rate)}</td>
        <td class="num">${s.mean_lead_time_w.toFixed(1)} wk</td>
        <td class="num">${s.mean_delay_w.toFixed(1)} wk</td>
        <td><div class="bar"><div class="track"><div class="bar-fill ${tone}" style="width:${s.risk_score}%"></div></div>
            <span>${s.risk_score.toFixed(0)}</span></div></td>
      </tr>`;
    })
    .join("");
}

// ------------------------------------------------------------------ charts
async function selectSku(sku: string) {
  selectedSku = sku;
  document.querySelectorAll("#stockout-table tbody tr").forEach((tr) =>
    tr.classList.toggle("selected", (tr as HTMLElement).dataset.sku === sku),
  );
  $("#forecast-title").textContent = `Demand forecast · ${sku}`;
  const data = await api.forecast(sku);
  const labels = [...data.history, ...data.forecast].map((p) => p.week);
  const pad = (n: number) => Array<number | null>(n).fill(null);
  const nh = data.history.length;
  // connect the forecast line to the last observed point
  const last = data.history[nh - 1].qty;

  const ds = [
    { label: "Actual", data: [...data.history.map((p) => p.qty), ...pad(data.forecast.length)],
      borderColor: css("--ink-2"), backgroundColor: css("--ink-2"), borderWidth: 1.5, pointRadius: 0 },
    { label: "p90", data: [...pad(nh - 1), last, ...data.forecast.map((p) => p.p90)],
      borderColor: "transparent", backgroundColor: css("--accent-soft"), pointRadius: 0, fill: "+1" },
    { label: "p10", data: [...pad(nh - 1), last, ...data.forecast.map((p) => p.p10)],
      borderColor: "transparent", pointRadius: 0, fill: false },
    { label: "Forecast", data: [...pad(nh - 1), last, ...data.forecast.map((p) => p.qty)],
      borderColor: css("--accent"), backgroundColor: css("--accent"), borderWidth: 2, pointRadius: 0 },
  ];

  forecastChart?.destroy();
  forecastChart = new Chart($<HTMLCanvasElement>("#forecast-chart"), {
    type: "line",
    data: { labels, datasets: ds },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: css("--ink-2"), filter: (i) => i.text === "Actual" || i.text === "Forecast" } },
        tooltip: { filter: (i) => i.raw !== null },
      },
      scales: {
        x: { ticks: { color: css("--ink-3"), maxTicksLimit: 8 }, grid: { display: false } },
        y: { ticks: { color: css("--ink-3") }, grid: { color: css("--grid") }, beginAtZero: true,
             title: { display: true, text: "units / week", color: css("--ink-3") } },
      },
    },
  });
}

function renderBacktest(models: Record<string, { wape: number }>) {
  const entries = Object.entries(models).sort((a, b) => a[1].wape - b[1].wape);
  new Chart($<HTMLCanvasElement>("#backtest-chart"), {
    type: "bar",
    data: {
      labels: entries.map(([m]) => m.replace(/_/g, " ")),
      datasets: [{
        label: "WAPE",
        data: entries.map(([, v]) => +(v.wape * 100).toFixed(1)),
        backgroundColor: entries.map(([m]) => (m === "gbm" ? css("--accent") : css("--neutral-bar"))),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: (c) => `WAPE ${c.raw}%` } } },
      scales: {
        x: { ticks: { color: css("--ink-3"), callback: (v) => `${v}%` }, grid: { color: css("--grid") }, beginAtZero: true },
        y: { ticks: { color: css("--ink-2") }, grid: { display: false } },
      },
    },
  });
}

// ------------------------------------------------------------------ boot
async function init() {
  try {
    const [summary, stockouts, suppliers, backtest] = await Promise.all([
      api.summary(), api.stockouts(), api.suppliers(), api.backtest(),
    ]);
    supplierNames = Object.fromEntries(suppliers.map((s) => [s.supplier_id, s.name]));
    $("#asof").textContent = `Data as of week starting ${summary.as_of_week}`;
    $("#summary-text").textContent = summary.text;
    const shortfall = stockouts
      .filter((r) => r.risk_level === "high")
      .reduce((s, r) => s + r.expected_shortfall_value_eur, 0);
    renderKpis(summary.high_risk_skus, summary.medium_risk_skus, shortfall,
      summary.forecast_wape_backtest, backtest.gbm_wape_reduction_vs_best_baseline, backtest.best_baseline);
    renderStockouts(stockouts);
    renderSuppliers(suppliers);
    renderBacktest(backtest.models);
    if (stockouts.length) await selectSku(stockouts[0].sku_id);

    document.querySelectorAll<HTMLButtonElement>(".filters button").forEach((b) =>
      b.addEventListener("click", async () => {
        document.querySelectorAll(".filters button").forEach((x) => x.classList.remove("active"));
        b.classList.add("active");
        renderStockouts(await api.stockouts((b.dataset.level || undefined) as RiskLevel | undefined));
      }),
    );
  } catch (err) {
    $("#summary-text").textContent =
      `Could not reach the API (${(err as Error).message}). Start it with: uvicorn scrm.api:app`;
  }
}

init();
