from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    max_tokens=700,
    temperature=0.9
)


prompt = ChatPromptTemplate.from_template("""
    You are FinSight AI, an AI financial research assistant.

    Your job is to analyse stocks using:
    1. Current market data
    2. Recent financial/news information
    3. The conversation history

    IMPORTANT:
    - Use the current ticker as the primary company.
    - Do not mix information from another company.
    - Financial data is the primary quantitative evidence.
    - News is supporting context.
    - If the available data is insufficient, say so.
    - Do not invent financial information.
    - Do not present your response as guaranteed financial advice.

    CURRENT TICKER:
    {ticker}

    CURRENT MARKET DATA:
    {market_data}

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
    Explain the main reasoning.

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
    """)


chain = prompt | llm | StrOutputParser()


def analyse_stock(
    ticker,
    question,
    market_data,
    context,
    chat_history
):

    result = chain.invoke({
        "ticker": ticker,
        "market_data": market_data,
        "context": context,
        "chat_history": chat_history,
        "question": question
    })

    return result