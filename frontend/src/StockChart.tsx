import { useEffect, useState } from "react";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, LoaderCircle } from "lucide-react";
import "./StockChart.css";

type Period = "1mo" | "3mo" | "6mo" | "1y";
type Point = { date: string; close: number };
type History = { ticker: string; period: Period; currency: string; prices: Point[] };
const periods: { label: string; value: Period }[] = [
  { label: "1M", value: "1mo" }, { label: "3M", value: "3mo" },
  { label: "6M", value: "6mo" }, { label: "1Y", value: "1y" },
];

export default function StockChart({ ticker }: { ticker: string }) {
  const [period, setPeriod] = useState<Period>("1mo");
  const [data, setData] = useState<History | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    async function loadHistory() {
      setLoading(true);
      setData(null);
      setError("");
      try {
        const response = await fetch(`/api/stock/${encodeURIComponent(ticker)}/history?period=${period}`, {
          signal: controller.signal,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "Unable to load prices");
        if (!Array.isArray(result.prices) || result.prices.length === 0) throw new Error("No price history available");
        if (!controller.signal.aborted) setData(result as History);
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : "Unable to load prices");
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void loadHistory();
    return () => controller.abort();
  }, [ticker, period]);

  const prices = data?.prices || [];
  const first = prices[0]?.close;
  const last = prices[prices.length - 1]?.close;
  const change = first && last ? ((last - first) / first) * 100 : null;
  const up = change !== null && change >= 0;
  const color = up ? "#36d39c" : "#fa7086";
  const formatPrice = (value: number) => `${data?.currency === "USD" ? "$" : ""}${value.toFixed(2)}`;

  return (
    <section className="stock-chart-card">
      <div className="stock-chart-top">
        <div>
          <div className="stock-chart-eyebrow">HISTORICAL MARKET DATA</div>
          <h3><Activity size={18} /> {ticker} Price History</h3>
          {change !== null && <span className={up ? "stock-chart-up" : "stock-chart-down"}>{up ? "+" : ""}{change.toFixed(2)}% over selected period</span>}
        </div>
        <div className="stock-chart-periods" aria-label="Chart time range">
          {periods.map((p) => <button type="button" key={p.value} className={period === p.value ? "active" : ""} onClick={() => setPeriod(p.value)}>{p.label}</button>)}
        </div>
      </div>
      {loading ? <div className="stock-chart-message"><LoaderCircle size={22} className="spinner" /> Loading price history...</div> :
        error ? <div className="stock-chart-message" role="alert">{error}</div> :
        <div className="stock-chart-plot" role="img" aria-label={`${ticker} historical daily closing price chart`}>
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={prices} margin={{ top: 14, right: 12, left: 0, bottom: 0 }}>
              <defs><linearGradient id="stockAreaFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity={0.25} /><stop offset="100%" stopColor={color} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid vertical={false} stroke="#28374b" strokeDasharray="3 5" />
              <XAxis dataKey="date" tick={{ fill: "#8596ad", fontSize: 11 }} tickFormatter={(d: string) => d.slice(5)} minTickGap={30} axisLine={false} tickLine={false} />
              <YAxis domain={["auto", "auto"]} tick={{ fill: "#8596ad", fontSize: 11 }} tickFormatter={(v: number) => v.toFixed(0)} width={48} axisLine={false} tickLine={false} />
              <Tooltip formatter={(value) => typeof value === "number" ? formatPrice(value) : String(value)} contentStyle={{ background: "#18263b", border: "1px solid #344764", borderRadius: 9, color: "#fff" }} labelStyle={{ color: "#c1cee3" }} />
              <Area type="monotone" dataKey="close" stroke={color} strokeWidth={2.5} fill="url(#stockAreaFill)" dot={false} activeDot={{ r: 5 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>}
      <p className="stock-chart-note">Daily adjusted closing prices via Yahoo Finance. Not real-time quotes.</p>
    </section>
  );
}
