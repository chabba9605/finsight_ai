from sklearn.ensemble import RandomForestClassifier


def create_model():
    """
    Create the machine learning model used by FinSight.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    return model