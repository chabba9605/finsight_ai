# FinSight AI 🔍📈

> **AI-powered financial research agent** — ask any question about a stock and get a grounded, real-time analysis in seconds.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?style=flat-square&logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-purple?style=flat-square)
![Tavily](https://img.shields.io/badge/Tavily-RAG-orange?style=flat-square)
![NVIDIA](https://img.shields.io/badge/NVIDIA-Nebius-76B900?style=flat-square&logo=nvidia)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=flat-square&logo=docker)
![Hackathon](https://img.shields.io/badge/Nebius_×_NVIDIA-Global_AI_Hackathon_2026-green?style=flat-square)

---

## What is FinSight AI?

FinSight AI answers natural language financial research questions by combining three sources of intelligence in a single API call:

- **Live market data** — current price, 5-day return, and key fundamentals via yFinance
- **Real-time news** — semantically relevant financial news retrieved live via Tavily RAG
- **Machine learning prediction** — a RandomForest classifier trained on 5 years of OHLCV data across 10 major tickers predicts next-day price direction

A three-agent LangGraph pipeline orchestrates all of this and passes the combined context to an NVIDIA Llama model hosted on Nebius, which returns a structured BUY / HOLD / SELL analysis with confidence rating, key financial factors, news impact, and risk assessment.

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│   FastAPI   │  POST /chat  or  POST /analyse
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────┐
│                  LangGraph Pipeline              │
│                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌──────────────────┐
│  │  Agent 1    │──▶│   Agent 2    │──▶│    Agent 3       │
│  │ Data Agent  │   │  Retrieval   │   │  Analysis Agent  │
│  │             │   │   Agent      │   │                  │
│  │ • yFinance  │   │ • Tavily RAG │   │ • NVIDIA LLM     │
│  │ • ML Model  │   │ • Live news  │   │ • Parse output   │
│  │ • RF Predict│   │ • Sources    │   │ • Update history │
│  └─────────────┘   └──────────────┘   └──────────────────┘
│                                                 │
└─────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ JSON Output │  recommendation · ml_prediction · sources · analysis
└─────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | NVIDIA Llama 3.1 70B via Nebius Token Factory |
| Agent Orchestration | LangGraph (StateGraph, MemorySaver) |
| LLM Chain | LangChain LCEL |
| News Retrieval (RAG) | Tavily API (real-time web search) |
| Market Data | yFinance |
| ML Prediction | scikit-learn RandomForestClassifier |
| API Framework | FastAPI + Uvicorn |
| Data Validation | Pydantic v2 |
| Containerisation | Docker + docker-compose |

---

## Example API Call

**Request:**
```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-001",
    "ticker": "NVDA",
    "message": "Should I buy NVDA this week?"
  }'
```

**Response:**
```json
{
  "session_id": "demo-001",
  "ticker": "NVDA",
  "message": "Should I buy NVDA this week?",
  "market_data": {
    "ticker": "NVDA",
    "current_price": 233.95,
    "five_day_return": 2.22
  },
  "ml_prediction": {
    "ticker": "NVDA",
    "prediction": "UP",
    "probability_up": 0.71,
    "probability_down": 0.29,
    "score": 0.71
  },
  "analysis": {
    "recommendation": "BUY",
    "confidence": "HIGH",
    "reason": "NVIDIA demonstrates exceptional fundamental strength with revenue growth of 83.4% and net income growth of 122.7%. The ML model predicts upward movement with 71% confidence, supported by strong institutional backing.",
    "key_financial_factors": [
      "Revenue (TTM) of $302.97B, up 83.4% year-over-year",
      "Forward P/E of 19.15 vs Trailing P/E of 29.19",
      "Net income growth of 122.7% with EPS at $7.91"
    ],
    "news_impact": [
      "Positive: Morgan Stanley reinstated NVDA as a Top Pick",
      "Positive: Amazon offloading $8B in Nvidia chips signals broad demand"
    ],
    "main_risk": "Intensifying competition from specialised AI chip competitors could erode pricing power.",
    "data_limitation": "None significant."
  },
  "sources": [
    "https://finance.yahoo.com/news/nvidia-earnings-2026",
    "https://www.reuters.com/technology/nvidia-morgan-stanley"
  ]
}
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker and docker-compose
- API keys for: Nebius Token Factory, Tavily, Groq (fallback)

### 1. Clone the repository

```bash
git clone https://github.com/chabba9605/finsight_ai.git
cd finsight_ai
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and add your keys:

```
GROQ_API_KEY=your-groq-key-here
NEBIUS_API_KEY=your-nebius-key-here
TAVILY_API_KEY=your-tavily-key-here
```

### 3. Train the ML model (first time only)

```bash
pip install -r requirements.txt
python -m ml.train
```

This downloads 5 years of OHLCV data for 10 major tickers and saves the trained model to `ml/stock_model.pkl`. Takes approximately 2 minutes.

### 4. Run with Docker

```bash
docker-compose up --build
```

### 5. Access the API

- **Interactive docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health
- **Analysis endpoint:** `POST http://localhost:8000/analyse`
- **Chat endpoint:** `POST http://localhost:8000/chat`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service health check |
| POST | `/chat` | Conversational analysis with session memory |
| POST | `/analyse` | Single-shot stock analysis |

### Request body (both endpoints)

```json
{
  "session_id": "unique-session-id",
  "ticker": "AAPL",
  "message": "Is Apple a good long-term hold?"
}
```

---

## Project Structure

```
finsight_ai/
├── main.py                  # FastAPI application entry point
├── graph.py                 # LangGraph pipeline definition
├── schema.py                # Pydantic request/response models
├── components/
│   ├── analyst.py           # LangChain + LLM analysis chain
│   └── state.py             # FinSightState TypedDict
├── nodes/
│   ├── data_agent.py        # Agent 1: yFinance + ML prediction
│   ├── retrieval_agent.py   # Agent 2: Tavily news retrieval
│   └── analysis_agent.py    # Agent 3: LLM analysis + parsing
├── ml/
│   ├── model.py             # RandomForest model + feature engineering
│   ├── train.py             # Multi-ticker training script
│   ├── predict.py           # Inference wrapper
│   └── stock_model.pkl      # Trained model artifact
├── rag/
│   └── retriever.py         # Tavily RAG client
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## ML Model

The prediction layer uses a **RandomForest Classifier** trained on 5 years of daily OHLCV data across 10 major tickers: AAPL, NVDA, MSFT, TSLA, GOOGL, AMZN, META, NFLX, AMD, and INTC.

**Features:**
- `return_1d` — 1-day price return
- `return_5d` — 5-day price return
- `ma_ratio` — 5-day MA / 20-day MA ratio
- `volatility` — 10-day rolling volatility
- `volume_change` — volume change vs previous day

**Training approach:** Chronological 80/20 split (no shuffling) to prevent data leakage from future into training set.

The ML prediction is passed to the LLM as an explicit input, allowing the model to incorporate the quantitative signal into its reasoning alongside fundamental data and news context.

---

## Hackathon

This project was built for the **Nebius × NVIDIA Global AI Hackathon 2026**.

- **Infrastructure:** Nebius Token Factory (NVIDIA Llama 3.1 70B)
- **RAG Partner:** Tavily real-time web search
- **Prize pool:** $50,000

---

## Author

**Raghav Chabba**
B.Sc. Computer Science — Richmond, The American University London

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Raghav_Chabba-blue?style=flat-square&logo=linkedin)](https://linkedin.com/in/raghavchabba)
[![GitHub](https://img.shields.io/badge/GitHub-chabba9605-black?style=flat-square&logo=github)](https://github.com/chabba9605)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
