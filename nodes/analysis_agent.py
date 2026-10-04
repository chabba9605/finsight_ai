from components.analyst import analyse_stock


def parse_analysis(text: str) -> dict:

    sections = {
        "recommendation": "",
        "confidence": "",
        "reason": "",
        "key_financial_factors": [],
        "news_impact": [],
        "main_risk": "",
        "data_limitation": ""
    }

    current_section = None

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line.startswith("RECOMMENDATION:"):
            sections["recommendation"] = (
                line.split(":", 1)[1].strip()
            )

        elif line.startswith("CONFIDENCE:"):
            sections["confidence"] = (
                line.split(":", 1)[1].strip()
            )

        elif line == "REASON:":
            current_section = "reason"

        elif line == "KEY FINANCIAL FACTORS:":
            current_section = "key_financial_factors"

        elif line == "NEWS IMPACT:":
            current_section = "news_impact"

        elif line == "MAIN RISK:":
            current_section = "main_risk"

        elif line == "DATA LIMITATION:":
            current_section = "data_limitation"

        elif current_section == "reason":
            sections["reason"] += (
                (" " if sections["reason"] else "")
                + line
            )

        elif current_section == "key_financial_factors":

            if line.startswith("-"):
                sections[
                    "key_financial_factors"
                ].append(line[1:].strip())

        elif current_section == "news_impact":

            if line.startswith("-"):
                sections[
                    "news_impact"
                ].append(line[1:].strip())

        elif current_section == "main_risk":

            sections["main_risk"] += (
                (" " if sections["main_risk"] else "")
                + line
            )

        elif current_section == "data_limitation":

            sections["data_limitation"] += (
                (" " if sections["data_limitation"] else "")
                + line
            )

    return sections


def analysis_agent(state):

    ticker = state["ticker"]
    question = state["question"]

    market_data = state["market_data"]

    context = state[
        "retrieved_context"
    ]

    chat_history = state.get(
        "chat_history",
        []
    )

    ml_score = state.get(
        "ml_score"
    )

    raw_analysis = analyse_stock(

        ticker=ticker,

        question=question,

        market_data=market_data,

        context=context,

        chat_history=chat_history,

        ml_score=ml_score
    )

    parsed_analysis = parse_analysis(
        raw_analysis
    )

    updated_history = chat_history + [
        {
            "role": "user",
            "content": question
        },
        {
            "role": "assistant",
            "content": raw_analysis
        }
    ]

    return {
        "analysis": parsed_analysis,
        "chat_history": updated_history
    }