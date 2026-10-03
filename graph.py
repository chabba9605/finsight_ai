from langgraph.graph import StateGraph, START, END
from components.state import FinSightState
from nodes.data_agent import data_agent
from nodes.retrieval_agent import retrieval_agent
from nodes.analysis_agent import analysis_agent

builder = StateGraph(FinSightState)
builder.add_node("data_agent", data_agent)
builder.add_node("retrieval_agent", retrieval_agent)
builder.add_node("analysis_agent", analysis_agent)

builder.add_edge(START, "data_agent")
builder.add_edge("data_agent", "retrieval_agent")
builder.add_edge("retrieval_agent", "analysis_agent")
builder.add_edge("analysis_agent", END)

pipeline = builder.compile()

#pipeline.invoke({"ticker": "NVDA", "question": "Should I buy this week?", "market_data": None, "retrieved_context": None, "sources": None, "analysis": None, "ml_score": None})