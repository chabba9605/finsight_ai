import { useEffect, useRef, useState, type FormEvent } from "react";
import {
  Activity, ArrowDownRight, ArrowUpRight, BrainCircuit, ExternalLink,
  History, LoaderCircle, Plus, Search, ShieldAlert, Sparkles, TrendingUp,
} from "lucide-react";
import "./App.css";
import "./chat.css";
import "./stream.css";
import "./citations.css";
import "./reliability.css";
import StockChart from "./StockChart";

type Analysis = {
  recommendation?: string;
  confidence?: string;
  reason?: string;
  key_financial_factors?: string[];
  news_impact?: string[];
  main_risk?: string;
  data_limitation?: string;
};

type StockResponse = {
  ticker: string;
  session_id: string;
  market_data?: { current_price?: number; five_day_return?: number };
  ml_prediction?: {
    prediction?: string;
    probability_up?: number;
    probability_down?: number;
    score?: number;
  };
  analysis?: Analysis;
  sources?: string[];
};

type ResearchTurn = {
  id: string;
  ticker: string;
  question: string;
  response: StockResponse;
  timestamp: string;
};

const SESSION_KEY = "finsight-session";
const historyKey = (id: string) => `finsight-history-${id}`;
const suggestions = ["NVDA", "AAPL", "MSFT", "TSLA", "GOOGL"];

function getSessionId(): string {
  let id = sessionStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function getHistory(id: string): ResearchTurn[] {
  try {
    const stored = JSON.parse(sessionStorage.getItem(historyKey(id)) || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch {
    return [];
  }
}

// Convert only [Source N] markers into links to the matching API source.
// Everything else remains ordinary React text (never injected HTML).
function CitedText({ text, sources = [] }: { text: string; sources?: string[] }) {
  const parts = text.split(/(\[Source\s+\d+\])/gi);
  return <>{parts.map((part, index) => {
    const match = /^\[Source\s+(\d+)\]$/i.exec(part);
    if (!match) return <span key={index}>{part}</span>;
    const number = Number(match[1]);
    const source = sources[number - 1];
    if (!source) return <span key={index}>{part}</span>;
    try {
      const url = new URL(source);
      if (url.protocol !== "https:" && url.protocol !== "http:") {
        return <span key={index}>{part}</span>;
      }
      return <a
        key={index}
        href={url.href}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-citation"
        title={`Open Source ${number}: ${url.hostname}`}
        aria-label={`Open source ${number} in a new tab`}
      >{part}<ExternalLink size={12} aria-hidden="true" /></a>;
    } catch {
      return <span key={index}>{part}</span>;
    }
  })}</>;
}

function App() {
  const [sessionId, setSessionId] = useState(getSessionId);
  const [history, setHistory] = useState<ResearchTurn[]>(() => getHistory(sessionId));
  const [activeTurnId, setActiveTurnId] = useState<string | null>(() => {
    const turns = getHistory(sessionId);
    return turns.at(-1)?.id ?? null;
  });
  const [ticker, setTicker] = useState("NVDA");
  const [question, setQuestion] = useState("Should I buy this stock based on recent market developments?");
  const [loading, setLoading] = useState(false);
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [error, setError] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const cancelledRef = useRef(false);
  const timedOutRef = useRef(false);
  const [serverStatus, setServerStatus] = useState<
    "starting" | "ready" | "unavailable"
    >("starting");

    useEffect(() => {
      const controller = new AbortController();
      const apiUrl = import.meta.env.VITE_API_URL || "/api";

      async function checkServer() {
        setServerStatus("starting");

        for (let attempt = 0; attempt < 12; attempt++) {
          if (controller.signal.aborted) return;

          try {
            const response = await fetch(`${apiUrl}/health`, {
              signal: controller.signal,
              cache: "no-store",
            });

            if (response.ok) {
              setServerStatus("ready");
              return;
            }
          } catch {
            if (controller.signal.aborted) return;
          }

          await new Promise<void>((resolve) => {
            setTimeout(resolve, 5000);
          });
        }

        if (!controller.signal.aborted) {
          setServerStatus("unavailable");
        }
      }

      void checkServer();

      return () => controller.abort();
    }, []);

  useEffect(() => () => {
    abortRef.current?.abort();
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
  }, []);

  function cancelAnalysis() {
    cancelledRef.current = true;
    abortRef.current?.abort();
  }

  useEffect(() => {
    sessionStorage.setItem(historyKey(sessionId), JSON.stringify(history));
  }, [history, sessionId]);

  const activeTurn = history.find((turn) => turn.id === activeTurnId);
  const result = activeTurn?.response;
  const market = result?.market_data;
  const ml = result?.ml_prediction;
  const analysis = result?.analysis;

  function startNewSession() {
    if (loading) return;
    const id = crypto.randomUUID();
    sessionStorage.setItem(SESSION_KEY, id);
    setSessionId(id);
    setHistory([]);
    setActiveTurnId(null);
    setTicker("NVDA");
    setQuestion("Should I buy this stock based on recent market developments?");
    setError("");
  }

  async function analyseStock(event?: FormEvent<HTMLFormElement>) {
    event?.preventDefault();
    const symbol = ticker.trim().toUpperCase();
    if (!symbol || loading) return;
    const prompt = question.trim() || `Analyse ${symbol} stock`;
    setLoading(true);
    setError("");

    setCompletedStages([]);
    cancelledRef.current = false;
    timedOutRef.current = false;
    const controller = new AbortController();
    abortRef.current = controller;
    // Protect the demo from a stalled connection. This is a total request limit.
    timeoutRef.current = setTimeout(() => {
      timedOutRef.current = true;
      controller.abort();
    }, 120_000);
    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL || "/api"}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({ session_id: sessionId, ticker: symbol, message: prompt }),
      });
      if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`);
      }
      if (!response.body) throw new Error("Streaming isn't supported by this browser.");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let report: StockResponse | null = null;
      let serverError = "";
      let finished = false;

      while (!finished) {
        const { done, value } = await reader.read();
        finished = done;
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.trim()) continue;
          const event = JSON.parse(line) as {
            type: string; stage?: string; result?: StockResponse; message?: string;
          };
          if (event.type === "stage_completed" && event.stage) {
            const stage = event.stage;
            setCompletedStages((current) => current.includes(stage)
              ? current : [...current, stage]);
          }
          if (event.type === "completed" && event.result) report = event.result;
          if (event.type === "error") serverError = event.message || "Analysis failed";
        }
      }
      if (serverError) throw new Error(serverError);
      if (!report) throw new Error("The connection ended before the report was complete. Please try again.");

      const turn: ResearchTurn = {
        id: crypto.randomUUID(),
        ticker: report.ticker || symbol,
        question: prompt,
        response: report,
        timestamp: new Date().toISOString(),
      };
      setHistory((previous) => [...previous, turn]);
      setActiveTurnId(turn.id);
      setQuestion("");
    } catch (err) {
      if (cancelledRef.current) {
        setError("Analysis cancelled. Your previous reports have been kept.");
      } else if (timedOutRef.current) {
        setError("The analysis took more than 2 minutes. Check the backend and try again.");
      } else if (err instanceof TypeError) {
        setError("Unable to connect to the FinSight AI server. The service may be starting up or temporarily unavailable. Please try again.");
      } else {
        setError(err instanceof Error ? err.message : "Unable to analyse this stock. Please retry.");
      }
    } finally {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
      abortRef.current = null;
      setLoading(false);
    }
  }

  const money = (n?: number) => typeof n === "number"
    ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(n)
    : "—";
  const percent = (n?: number) => typeof n === "number" ? `${(n * 100).toFixed(1)}%` : "—";
  const return5d = market?.five_day_return;
  const positive = typeof return5d === "number" && return5d >= 0;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon"><TrendingUp size={23} /></div>
          <div><strong>FinSight<span> AI</span></strong><small>Financial intelligence</small></div>
        </div>
        <div className="sidebar-label">WORKSPACE</div>
        <div className="nav-active"><Activity size={18} /> Research dashboard</div>
        <button className="new-session-button" onClick={startNewSession} disabled={loading}>
          <Plus size={17} /> New session
        </button>
        <div className="history-heading"><History size={15} /> CONVERSATION HISTORY</div>
        <div className="history-list">
          {history.length === 0 && <p className="history-empty">No research in this session yet.</p>}
          {[...history].reverse().map((turn) => (
            <button
              key={turn.id}
              className={`history-entry ${activeTurnId === turn.id ? "history-entry-active" : ""}`}
              onClick={() => {
                setActiveTurnId(turn.id);
                setTicker(turn.ticker);
                setError("");
              }}
              title={turn.question}
            >
              <strong>{turn.ticker}</strong>
              <span>{turn.question}</span>
            </button>
          ))}
        </div>
        <div className="sidebar-bottom">
          <div className={`backend-status ${serverStatus}`}>
            <span className="backend-status-dot" />
            {serverStatus === "ready" 
            ? "AI Server Online" 
            : serverStatus === "starting" ? "Starting AI Server..." 
            : "AI Server Unavailable"}
          </div>
          <p>Session: <code title={sessionId}>{sessionId.slice(0, 8)}...</code></p>
          <button className="session-copy" onClick={() => void navigator.clipboard.writeText(sessionId)}>
            Copy session ID
          </button>
          <p>Session memory resets when the backend restarts.</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <span className="eyebrow">FINANCIAL RESEARCH TERMINAL</span>
            <h1>Market Intelligence</h1>
            <p>Explore stocks with live data, machine learning and AI research.</p>
          </div>
          <div className="topbar-pill"><Sparkles size={15} /> Powered by Nebius</div>
        </header>

        <section className="research-panel">
          <div className="panel-heading">
            <div><h2>Research a stock</h2><p>Ask a new question or follow up in the current session.</p></div>
            <BrainCircuit size={24} className="muted-icon" />
          </div>
          <form onSubmit={analyseStock}>
            <div className="input-row">
              <div className="ticker-field">
                <Search size={18} />
                <input aria-label="Stock ticker" value={ticker}
                  onChange={(e) => setTicker(e.target.value.toUpperCase())}
                  placeholder="Enter ticker, e.g. NVDA" maxLength={12} />
              </div>
              <button className="analyse-button" disabled={loading}>
                {loading ? <><LoaderCircle size={17} className="spinner" /> Analysing...</>
                  : <><Sparkles size={17} /> Analyse stock</>}
              </button>
            </div>
            <textarea aria-label="Research question" value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a follow-up question..." rows={3} />
            <div className="quick-tickers"><span>POPULAR</span>
              {suggestions.map((symbol) => (
                <button key={symbol} type="button" className={ticker === symbol ? "chip selected" : "chip"}
                  onClick={() => setTicker(symbol)}>{symbol}</button>
              ))}
            </div>
          </form>
        </section>

        {error && <div className="error-banner" role="alert">
          <ShieldAlert size={18} aria-hidden="true" />
          <span>{error}</span>
          <button type="button" className="dismiss-error" onClick={() => setError("")}>Dismiss</button>
        </div>}
        {loading && <section className="stream-progress" aria-live="polite">
          <div className="stream-progress-top">
            <h2><LoaderCircle size={20} className="spinner accent" /> Running financial research</h2>
            <button type="button" className="cancel-analysis" onClick={cancelAnalysis}>Cancel</button>
          </div>
          {[
            ["market_data", "Market Data Agent", "Stock prices and ML prediction"],
            ["news_retrieval", "Retrieval Agent", "Tavily news and sources"],
            ["ai_analysis", "Analysis Agent", "Nebius research report"],
          ].map(([id, label, description], i) => {
            const done = completedStages.includes(id);
            const active = !done && (i === 0 || completedStages.includes(
              ["market_data", "news_retrieval"][i - 1]
            ));
            return <div className="stream-stage" key={id}>
              <span className={`stream-dot ${done ? "done" : active ? "active" : ""}`}>
                {done ? "✓" : active ? "●" : "○"}
              </span>
              <div><strong>{label}</strong><small>{description}</small></div>
              <span className="stream-status">{done ? "Completed" : active ? "Processing…" : "Waiting"}</span>
            </div>;
          })}
        </section>}
        {!result && !loading && !error && <div className="empty-state">
          <div className="empty-icon"><BrainCircuit size={32} /></div>
          <h2>Your research starts here</h2>
          <p>Ask a question to create your first research report. Follow-up questions will use the same session ID.</p>
        </div>}

        {result && !loading && (
          <div className="results">
            <div className="results-heading">
              <div>
                <span className="eyebrow">RESEARCH REPORT</span>
                <h2>{result.ticker} <span>Stock Analysis</span></h2>
              </div>
              <span className="report-label">AI generated</span>
            </div>
            <div className="question-banner"><span>YOUR QUESTION</span><p>{activeTurn?.question}</p></div>
            <StockChart ticker={result.ticker} />

            <div className="metrics-grid">
              <article className="metric-card"><span>MARKET PRICE</span><strong>{money(market?.current_price)}</strong><small>Latest available market price</small></article>
              <article className="metric-card"><span>5-DAY RETURN</span>
                <strong className={typeof return5d !== "number" ? "" : positive ? "positive" : "negative"}>
                  {typeof return5d === "number" ? <>
                    {positive ? <ArrowUpRight size={23} /> : <ArrowDownRight size={23} />}
                    {return5d.toFixed(2)}%</> : "—"}
                </strong><small>Recent price performance</small>
              </article>
              <article className="metric-card"><span>AI RECOMMENDATION</span>
                <strong className="recommendation">{analysis?.recommendation || "—"}</strong>
                <small>Confidence: {analysis?.confidence || "Not available"}</small>
              </article>
              <article className="metric-card"><span>ML DIRECTION</span>
                <strong>{ml?.prediction || "—"}</strong><small>Next-day model prediction</small>
              </article>
            </div>
            <div className="details-grid">
              <article className="content-card">
                <div className="card-title"><BrainCircuit size={19} /><h3>AI Research Summary</h3></div>
                <p className="analysis-text"><CitedText text={analysis?.reason || "No research summary returned."} sources={result.sources} /></p>
                <h4>Key financial factors</h4>
                <ul>{(analysis?.key_financial_factors || []).map((item, i) => <li key={i}><CitedText text={item} sources={result.sources} /></li>)}</ul>
              </article>
              <article className="content-card">
                <div className="card-title"><Activity size={19} /><h3>Machine Learning Signal</h3></div>
                <div className="probability-label"><span>Probability UP</span><strong>{percent(ml?.probability_up)}</strong></div>
                <div className="progress-track"><div className="progress-fill green" style={{width: `${Math.max(0, Math.min(100, (ml?.probability_up || 0) * 100))}%`}} /></div>
                <div className="probability-label second"><span>Probability DOWN</span><strong>{percent(ml?.probability_down)}</strong></div>
                <div className="progress-track"><div className="progress-fill red" style={{width: `${Math.max(0, Math.min(100, (ml?.probability_down || 0) * 100))}%`}} /></div>
                <p className="model-note">Random Forest directional estimates. The model has not demonstrated reliable predictive advantage over chance.</p>
              </article>
              <article className="content-card">
                <div className="card-title"><TrendingUp size={19} /><h3>News & Market Impact</h3></div>
                <ul>{(analysis?.news_impact || []).map((item, i) => <li key={i}><CitedText text={item} sources={result.sources} /></li>)}</ul>
                {!analysis?.news_impact?.length && <p className="secondary-text">No news impact summary available.</p>}
              </article>
              <article className="content-card">
                <div className="card-title"><ShieldAlert size={19} /><h3>Risks & Limitations</h3></div>
                <h4>Main risk</h4><p className="analysis-text"><CitedText text={analysis?.main_risk || "Not provided."} sources={result.sources} /></p>
                <h4>Data limitations</h4><p className="analysis-text"><CitedText text={analysis?.data_limitation || "Not provided."} sources={result.sources} /></p>
              </article>
            </div>
            <article className="content-card sources-card">
              <div className="card-title"><ExternalLink size={19} /><h3>Research Sources</h3></div>
              {result.sources?.length ? <div className="source-list">
                {result.sources.map((url, i) => {
                  let host = url;
                  try {
                    const parsed = new URL(url);
                    if (!["http:", "https:"].includes(parsed.protocol)) return null;
                    host = parsed.hostname.replace(/^www\./, "");
                  } catch { return null; }
                  return <a href={url} key={i} target="_blank" rel="noopener noreferrer" className="source-link">
                    <span>Source {i + 1} · {host}</span><ExternalLink size={16} />
                  </a>;
                })}
              </div> : <p className="secondary-text">No sources returned.</p>}
            </article>
          </div>
        )}
        <footer className="footer">FinSight AI · Research assistance only · Not investment advice</footer>
      </main>
    </div>
  );
}

export default App;
