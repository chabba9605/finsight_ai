from groq import Groq
from dotenv import load_dotenv
import os, yfinance as yf
from langchain_groq import *
from langchain_core.prompts import *
from langchain_core.output_parsers import *

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class FinancialAnalystChain():


    def analyse_stock(self,ticker: str, question: str) ->str:

        stock = yf.Ticker(ticker)
        
        price = stock.history(period="1d")["Close"].iloc[-1]

        prompt = ChatPromptTemplate.from_messages([
            ("system" , """
                You are FinSight, an AI stock analysis assistant.

                Your job is to analyse the stock using the financial data provided to you
                and give a clear, concise investment assessment.

                You MUST give one of these three conclusions:

                BUY
                HOLD
                SELL

                Do not avoid choosing one of these three options unless there is genuinely
                no usable data at all.

                Base your decision on the data provided, not on assumptions or made-up facts.

                Consider:
                - Current stock price
                - Recent price movement
                - Valuation
                - Earnings
                - Revenue growth
                - Profitability
                - Debt
                - Cash flow
                - Market trend
                - Risk
                - Any other financial metrics provided

                Your analysis should answer:

                1. What is the recommendation?
                2. Why?
                3. What are the 2-3 strongest factors supporting the recommendation?
                4. What is the biggest risk that could make this recommendation wrong?

                IMPORTANT:
                - Do not invent missing financial information.
                - If a metric is unavailable, ignore it rather than guessing.
                - Do not give generic financial disclaimers.
                - Do not repeat the entire dataset.
                - Do not give a long explanation.
                - Do not simply say "it depends".
                - Make a decision based on the available evidence.
                - The recommendation should be supported by specific data whenever possible.

                Return your answer EXACTLY in this structure:

                RECOMMENDATION: BUY / HOLD / SELL

                CONFIDENCE: HIGH / MEDIUM / LOW

                REASON:
                Give 2-4 sentences explaining the decision using the most important
                available financial data.

                KEY FACTORS:
                • Factor 1
                • Factor 2
                • Factor 3

                MAIN RISK:
                One short paragraph explaining the biggest risk to this assessment.

                DATA LIMITATION:
                Only mention this if important data is missing. Otherwise write:
                "None significant."

                Keep the entire response under 200 words.
                """),
            ("user", """
                Analyse the following stock:

                Ticker: {ticker}
                Current Price: {price}

                Investor Question:
                {question}

                Give your BUY, HOLD, or SELL assessment based only on the available data.

                """)
        ])

        chain = prompt | ChatGroq(model = "qwen/qwen3.8-27b") | StrOutputParser()

        result = chain.invoke({"ticker" : ticker, "price" : price, "question" : "Buy or sell" })

        return result


print(FinancialAnalystChain().analyse_stock("MCD","Should I buy or sell this stock based on its current price?"))