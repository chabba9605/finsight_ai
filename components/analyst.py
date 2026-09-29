from groq import Groq
from dotenv import load_dotenv
import os, yfinance as yf

load_dotenv()
client = Groq(api_key=os.getenv("Groq_API_KEY"))

def analyse_stock(ticker: str, question: str) -> str:
    stock = yf.Ticker(ticker)
    price = stock.history(period="1d")["Close"].iloc[-1]

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {"role": "system", "content": "You are a concise financial analyst whose main job is to tell people bluntly that if they should buy a stock or not no need for the deep explanation but do not say yes to every stock they ask about. Take your time analyse the history and make a careful judgment if it is above 70% chances of growth in the stock then only advise to invest. While responding mention only a few things such as Ticker, current price, reccomendation(yes/no) and the time-period and a brief Reason for either yes or no. Also advise on how much"},
            {"role": "user",   "content": f"Ticker: {ticker}. Price: £{price:.2f}. Question: {question}"}
        ]
    )
    return response.choices[0].message.content

print(analyse_stock("MCD", "Should I buy this stock for investment and for how much time respond in json"))