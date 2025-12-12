from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def build_boosting_model(random_state=42):
    base_estimator = DecisionTreeClassifier(max_depth=1, random_state=random_state)
    return AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=50,
        learning_rate=0.5,
        random_state=random_state
    )
