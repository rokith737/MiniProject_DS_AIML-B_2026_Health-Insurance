# 🛡️ Insurance Fraud Detection System
### MiniProject_DS_AIML-B_2026_Health Insurance

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)
![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 Project Title
**Multimodal Insurance Fraud Detection System using Ensemble Machine Learning and Explainable AI**

---

## 📝 Abstract

Insurance fraud is a critical financial threat causing billions of dollars in annual losses worldwide. This project presents a multimodal, AI-powered insurance fraud detection system trained on the publicly available CMS Medicare Healthcare Provider Fraud Detection dataset comprising over 700,000 real claim records. The system employs an ensemble of three machine learning models — XGBoost, LightGBM, and Random Forest — combined through soft voting to achieve a high-performance fraud classifier. An extensive feature engineering pipeline constructs 57 features across seven groups including ratio features, composite risk scores, interaction features, z-score deviations, anomaly flags, and quantile bins. SMOTE is applied to address severe class imbalance. The system integrates SHAP (SHapley Additive exPlanations) for transparent, per-prediction natural language explanations. A Flask-based web application with four functional tabs allows investigators to manually enter claim features or upload documents for OCR-based scanning. The system achieves an AUC-ROC above 0.90 and provides actionable risk tiers (CRITICAL / HIGH / MEDIUM / LOW) with investigator-ready fraud reports.

---

## ❗ Problem Statement

Healthcare insurance fraud — where providers submit false or inflated claims — costs Medicare and Medicaid billions annually. Traditional rule-based detection systems miss complex patterns such as fraud rings, diagnosis stuffing, and inflated billing cycles. Existing ML approaches output a single score with no explanation, making them unusable by human investigators. This project addresses all three gaps: it detects complex fraud patterns using ensemble ML, explains every prediction using SHAP, and provides a document-aware interface that works directly on uploaded claim PDFs and images.

---

## 📊 Dataset Source

| Property | Details |
|---|---|
| Name | Healthcare Provider Fraud Detection Analysis |
| Source | [Kaggle — rohitrox](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis) |
| Size | 123.66 MB across 8 CSV files |
| Records | 700,000+ claim rows → 5,410 labeled providers |
| Type | Supervised binary classification (fraud / not fraud) |
| Files | Train.csv, Train_Beneficiarydata.csv, Train_Inpatientdata.csv, Train_Outpatientdata.csv + 4 Test equivalents |

---

## 🔬 Methodology / Workflow

```
1. Data Collection
   └── Download 8 CSV files from Kaggle CMS Medicare dataset

2. Data Merging & Preprocessing  (notebooks/preprocessing.ipynb)
   ├── Merge 4 Train CSVs on BeneID + Provider keys
   ├── Handle missing values (median/mode imputation)
   ├── Parse date features → duration features
   └── Aggregate claim rows → one row per Provider

3. Exploratory Data Analysis  (notebooks/data_understanding.ipynb)
   ├── Class imbalance analysis (~10% fraud)
   ├── Financial fraud signals (reimbursement gaps)
   ├── Patient volume & chronic condition patterns
   └── Correlation heatmap + SHAP feature ranking

4. Visualization  (notebooks/visualization.ipynb)
   ├── 8 publication-quality plots
   ├── Fraud rate by quantile bucket
   └── Feature importance (Mutual Information)

5. Feature Engineering  (src/preprocessing.py)
   ├── 57 features across 7 groups
   ├── Ratio, risk scores, interaction, z-scores, flags, bins
   └── SMOTE for class balancing

6. Model Development  (src/model.py)
   ├── Logistic Regression (baseline)
   ├── Random Forest (baseline tree)
   ├── XGBoost + Optuna tuning (50 trials)
   ├── LightGBM + Optuna tuning (50 trials)
   ├── Isolation Forest (anomaly layer)
   └── Soft Voting Ensemble (XGB×3 + LGBM×3 + RF×1)

7. Explainability  (src/analysis.py)
   ├── SHAP TreeExplainer
   ├── Per-prediction natural language reasons
   └── Investigator PDF report generation

8. Deployment  (flask_app/app.py)
   ├── Flask backend (4 endpoints)
   ├── Gradio/HTML frontend (4 tabs)
   └── Document OCR scanner (EasyOCR)
```

---

## 🛠️ Tools & Technologies

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy, PyArrow |
| Visualization | Matplotlib, Seaborn, Plotly |
| ML Models | XGBoost, LightGBM, Scikit-learn |
| Hyperparameter Tuning | Optuna (Bayesian optimization) |
| Class Balancing | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| Anomaly Detection | Isolation Forest, Keras Autoencoder |
| Graph Analysis | NetworkX |
| OCR | EasyOCR, pdf2image, Pillow |
| Web Framework | Flask 3.0 |
| Report Generation | ReportLab |
| Environment | Google Colab (T4 GPU), VS Code |

---

## 📈 Results / Findings

| Metric | Value |
|---|---|
| AUC-ROC (Validation) | > 0.90 |
| F1-Score (Fraud class) | > 0.82 |
| Precision | > 0.80 |
| Recall | > 0.78 |
| Features Engineered | 57 |
| Models in Ensemble | 3 (XGBoost + LightGBM + RF) |

**Key Findings from EDA:**
- Fraudulent providers bill **3–5× more** per claim than legitimate ones
- Average hospital stay for fraud: **40+ days** vs 5 days for legitimate
- Fraud providers use **2× more unique physicians** per claim
- High chronic condition scores (>5) strongly correlate with fraud
- Providers triggering 4+ anomaly flags have **>85% fraud probability**

---

## 👥 Team Members

| Name | Role | GitHub |
|---|---|---|
| [Swasthika S] | ML Model Training, Feature Engineering,  Data Preprocessing, EDA  | [https://github.com/SwasthikaSelvakumar] |
| [ROkith S] | Flask Deployment, UI, Documentation, Report| [@username] |

> **Department:** AIML 

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/[username]/MiniProject_DS_AIML-B_2026_InsuranceFraudDetection
cd MiniProject_DS_AIML-B_2026_InsuranceFraudDetection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle and place in dataset/raw_data/

# 4. Run notebooks in order:
#    notebooks/preprocessing.ipynb  → generates processed data
#    notebooks/data_understanding.ipynb → EDA
#    notebooks/visualization.ipynb → plots

# 5. Run Flask app (after training models via Colab)
cd flask_app
python app.py
# Open http://localhost:5000
```

---

## 📂 Repository Structure

```
MiniProject_DS_AIML-B_2026_InsuranceFraudDetection/
│
├── README.md
├── requirements.txt
│
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
│
├── dataset/
│   ├── raw_data/           ← Place Kaggle CSVs here
│   └── processed_data/     ← Merged parquet files saved here
│
├── notebooks/
│   ├── data_understanding.ipynb    ← M2: EDA
│   ├── preprocessing.ipynb         ← M1: Data merging & cleaning
│   └── visualization.ipynb         ← M2: All 8 plots
│
├── src/
│   ├── preprocessing.py    ← Feature engineering functions
│   ├── analysis.py         ← SHAP explainability functions
│   └── model.py            ← Model training & ensemble
│
├── outputs/
│   ├── graphs/             ← EDA + SHAP plots saved here
│   └── results/            ← Model metrics, comparison JSON
│
├── flask_app/              ← Web application
│   ├── app.py
│   ├── templates/index.html
│   └── models/             ← Trained .pkl files
│
└── report/
    └── mini_project_report.pdf
```

---

## 📄 License
MIT License — see [LICENSE](LICENSE) for details.
