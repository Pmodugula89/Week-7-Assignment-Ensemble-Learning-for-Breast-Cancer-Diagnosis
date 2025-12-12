# Week-7-Assignment-Ensemble-Learning-for-Breast-Cancer-Diagnosis
Objective
Explore ensemble modeling principles and their impact on performance and stability.
Implement and compare Bagging and Boosting to improve accuracy and reliability on a breast‑cancer diagnosis task.

Healthcare Scenario
You are a data scientist at MedTech Innovations.
Existing diagnostic models show promise but exhibit high variance and unstable predictions.
Your job is to apply Bagging and Boosting to produce more reliable outputs for oncologists.

Toolchain
GitHub — version control and submission
Visual Studio Code — local Python development (integrated terminal & debugger)
GitHub Copilot — AI‑assisted coding (used responsibly)
Open‑source libraries — NumPy, Pandas, scikit‑learn, Matplotlib, Seaborn
(Optional) XGBoost for an extra boosting baseline
Project Structure
Week-7-Assignment-Ensemble-Learning-for-Breast-Cancer-Diagnosis/ │ ├── README.md ├── requirements.txt ├── .gitignore │ ├── src/ │ ├── data_load.py │ ├── preprocess.py │ ├── base_tree.py │ ├── bagging.py │ ├── boosting.py │ ├── evaluate.py │ └── main.py │ ├── figures/ │ ├── confusion_matrix_base_tree.png │ ├── confusion_matrix_bagging.png │ └── confusion_matrix_boosting.png │ ├── ppt/ │ └── ensemble_diagnosis.pptx │ └── video/ └── ensemble_briefing.mp4

Dataset
Source: sklearn.datasets.load_breast_cancer()
Well‑known, clean dataset with no missing values.
Features: 30 numeric attributes describing cell nuclei.
Target: Binary classification (malignant vs benign).
Setup Instructions
Clone the repository

git clone https://github.com/Manishakittu/cst600-week07-ensembles
cd Week-7-Assignment-Ensemble-Learning-for-Breast-Cancer-Diagnosis


**Create and activate a virtual environment**
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate # macOS/Linux

**Install dependencies**
pip install -r requirements.txt

**Run the pipeline**
python src/main.py

**Workflow**
- Data Exploration & Preprocessing
- Load dataset, check class balance, scale features if needed.
- Pipeline ensures reproducibility.
- Baseline Model
- DecisionTreeClassifier (moderate depth).
- Bagging
- BaggingClassifier with Decision Tree base estimator.
- n_estimators=50, bootstrap=True, oob_score=True.
- Boosting
- AdaBoostClassifier with shallow trees (stumps).
- n_estimators=50, learning_rate=0.5.
- Evaluation
- Accuracy, Precision, Recall, F1, Confusion Matrix.
- Stratified K‑Fold CV for stability (mean ± std).
- Clinical Interpretation
- Improved recall → fewer missed positives.
- Bagging reduces variance; Boosting reduces bias.


**Example Output**
Console metrics (sample):
Base Tree Metrics:
Accuracy: 0.93
Precision: 0.92
Recall: 0.94
F1 Score: 0.93
Cross-Validated F1 Score: 0.92 ± 0.02

**References**
- scikit‑learn documentation: Decision Trees, BaggingClassifier, AdaBoostClassifier
- Neptune.ai — Mwiti, D. (2023). Ensemble Learning
- GeeksforGeeks (2022). Bagging vs Boosting
