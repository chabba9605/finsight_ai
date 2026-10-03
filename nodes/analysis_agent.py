from components.analyst import FinancialAnalystChain
from components.state import FinSightState

analyst = FinancialAnalystChain()

def analysis_agent(state: FinSightState) -> FinSightState:
    result = analyst.analyse_stock(
        ticker=state["ticker"],
        question=state["question"]
    )
    state["analysis"] = result
    return state