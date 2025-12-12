from sklearn.tree import DecisionTreeClassifier

def build_base_model(random_state=42):
    return DecisionTreeClassifier(max_depth=5, random_state=random_state)
