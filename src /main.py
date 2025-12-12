from data_load import load_data
from preprocess import build_preprocessing_pipeline
from base_tree import build_base_model
from bagging import build_bagging_model
from boosting import build_boosting_model
from evaluate import evaluate_model, plot_confusion_matrix, cross_validate_model

from sklearn.model_selection import train_test_split

def run_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = build_preprocessing_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    models = {
        'Base Tree': build_base_model(),
        'Bagging': build_bagging_model(),
        'Boosting': build_boosting_model()
    }

    for name, model in models.items():
        model.fit(X_train_transformed, y_train)
        metrics = evaluate_model(model, X_test_transformed, y_test)
        print(f"\n{name} Metrics:")
        for k, v in metrics.items():
            if k != 'Confusion Matrix':
                print(f"{k}: {v:.4f}")
        plot_confusion_matrix(metrics['Confusion Matrix'], f"{name} Confusion Matrix", f"confusion_matrix_{name.lower().replace(' ', '_')}.png")
        mean_f1, std_f1 = cross_validate_model(model, X_train_transformed, y_train)
        print(f"Cross-Validated F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")

if __name__ == "__main__":
    run_pipeline()
