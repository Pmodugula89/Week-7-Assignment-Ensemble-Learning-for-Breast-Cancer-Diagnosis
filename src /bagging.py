from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

def build_bagging_model(random_state=42):
    base_estimator = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    return BaggingClassifier(
        estimator=base_estimator,
        n_estimators=50,
        bootstrap=True,
        oob_score=True,
        random_state=random_state
    )
