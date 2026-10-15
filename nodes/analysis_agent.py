from components.analyst import analyse_stock
from components.state import FinSightState
import re



def parse_analysis(text: str) -> dict:
    sections = {
        "recommendation": "",
        "confidence": "",
        "reason": "",
        "key_financial_factors": [],
        "news_impact": [],
        "main_risk": "",
        "data_limitation": "",
    }

    headings = {
        "RECOMMENDATION": "recommendation",
        "CONFIDENCE": "confidence",
        "REASON": "reason",
        "KEY FINANCIAL FACTORS": "key_financial_factors",
        "NEWS IMPACT": "news_impact",
        "MAIN RISK": "main_risk",
        "DATA LIMITATION": "data_limitation",
    }

    current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        # Support plain text and Markdown-formatted headings.
        clean = line.strip("# ").replace("**", "").strip()

        match = re.match(
            r"^([A-Za-z ]+)\s*:\s*(.*)$",
            clean,
        )

        if match:
            heading = match.group(1).upper().strip()

            if heading in headings:
                current = headings[heading]
                value = match.group(2).strip()

                if value:
                    if isinstance(sections[current], list):
                        sections[current].append(value.lstrip("-• "))
                    else:
                        sections[current] = value

                continue

        if current is None:
            continue

        if isinstance(sections[current], list):
            value = clean.lstrip("-• ").strip()
            if value:
                sections[current].append(value)
        else:
            sections[current] += (
                (" " if sections[current] else "") + clean
            )


    
    # Require a single valid recommendation.
    recommendation = sections["recommendation"].strip().upper()
    confidence = sections["confidence"].strip().upper()

    if recommendation not in {"BUY", "HOLD", "SELL"}:
        raise ValueError(
            "The AI did not return a valid BUY, HOLD, "
            "or SELL recommendation. Please retry."
        )

    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        raise ValueError(
            "The AI did not return a valid confidence level. "
            "Please retry."
        )

    if not sections["reason"].strip():
        raise ValueError(
            "The AI did not explain its recommendation. "
            "Please retry."
        )

    sections["recommendation"] = recommendation
    sections["confidence"] = confidence

    return sections



def analysis_agent(
    state: FinSightState,
) -> FinSightState:
    """
    Generate the final LLM analysis and store it
    in the LangGraph state.
    """

    ticker = state["ticker"]

    question = state["question"]

    chat_history = state.get(
        "chat_history",
        [],
    )

    raw_analysis = analyse_stock(
        ticker=ticker,
        question=question,
        market_data=state.get(
            "market_data",
            {},
        ),
        ml_prediction=state.get(
            "ml_prediction"
        ),
        context=state.get(
            "retrieved_context",
            "",
        ),
        chat_history=chat_history,
    )

    parsed_analysis = parse_analysis(
        raw_analysis
    )

    # Update conversation history.
    updated_history = (
        chat_history
        + [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": raw_analysis,
            },
        ]
    )

    return {
        "analysis": parsed_analysis,
        "chat_history": updated_history,
    }