from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI 
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")

if not NEBIUS_API_KEY:
    raise ValueError(
        "NEBIUS_API_KEY is not set. "
        "Make sure your .env file contains:\n"
        "NEBIUS_API_KEY=your_api_key_here"
    )

llm = ChatOpenAI(
    model="zai-org/GLM-5.3",
    api_key=NEBIUS_API_KEY,
    base_url="https://api.studio.nebius.ai/v1/",
    temperature=0.9
)

prompt = ChatPromptTemplate.from_template(
    """
You are FinSight AI, an AI financial research assistant.

Your job is to analyse a stock using:

1. Current market data
2. Machine learning prediction
3. Recent financial/news information
4. Previous conversation history

IMPORTANT RULES:

- Use the current ticker as the primary company.
- Never mix information from another company.
- Financial data is the primary quantitative evidence.
- The ML prediction is a supporting quantitative signal.
- News is supporting context.
- Do not invent financial information.
- If information is unavailable, explicitly say so.
- Do not present your response as guaranteed financial advice.
- Treat the ML prediction as a probability-based signal, not certainty.
- Consider conflicting evidence rather than blindly following the ML model.

CURRENT TICKER:
{ticker}

CURRENT MARKET DATA:
{market_data}

MACHINE LEARNING PREDICTION:
{ml_prediction}

RECENT NEWS / RETRIEVED INFORMATION:
{context}

PREVIOUS CONVERSATION:
{chat_history}

CURRENT USER QUESTION:
{question}

Respond using exactly this structure:

RECOMMENDATION: BUY / HOLD / SELL

CONFIDENCE: HIGH / MEDIUM / LOW

REASON:
Explain the main reasoning and how the market data, ML signal and news contribute.

KEY FINANCIAL FACTORS:
- Factor 1
- Factor 2
- Factor 3

NEWS IMPACT:
- Impact 1
- Impact 2

MAIN RISK:
Explain the biggest risk.

DATA LIMITATION:
Mention important missing or uncertain information, or say "None significant."

Keep the response under 250 words.
"""
)


chain = (
    prompt
    | llm
    | StrOutputParser()
)


def analyse_stock(
    ticker,
    question,
    market_data,
    ml_prediction,
    context,
    chat_history,
):
    """
    Generate the final financial analysis using
    market data, ML prediction, retrieved information,
    and conversation history.
    """

    result = chain.invoke(
        {
            "ticker": ticker,
            "question": question,
            "market_data": market_data,
            "ml_prediction": ml_prediction,
            "context": context,
            "chat_history": chat_history,
        }
    )

    return result