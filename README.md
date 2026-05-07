# 🏥 Predicting Extubation Failure in ICU Patients
### A Machine Learning Approach Using MIMIC-IV Clinical Data

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange?logo=data:image/png;base64,)
![License](https://img.shields.io/badge/License-Academic-green)
![Dataset](https://img.shields.io/badge/Dataset-MIMIC--IV-red)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Clinical Background](#clinical-background)
- [Dataset](#dataset)
- [Project Pipeline](#project-pipeline)
- [Key Results](#key-results)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Team](#team)

---

## Overview

Extubation failure — the need to re-intubate a patient within 72 hours of removing a mechanical ventilator — occurs in **~8–20% of ICU patients** and is associated with significantly increased mortality, prolonged ICU stay, and higher healthcare costs.

This project builds a **clinical decision-support model** that predicts extubation failure risk using the last 24 hours of vital signs, blood gas measurements, and ventilator settings recorded before the extubation attempt. The model is trained on **24,305 ICU admissions** from the MIMIC-IV database.

---

## Clinical Background

### What is Extubation Failure?
Mechanical ventilation is a life-sustaining intervention in the ICU. When the clinical team decides a patient is ready to breathe independently, they remove the endotracheal tube — a process called **extubation**. If the patient cannot sustain adequate breathing, they require **re-intubation within 72 hours**, which is classified as extubation failure.

### Key Clinical Indicators Used
| Feature | Clinical Meaning |
|---------|-----------------|
| **RSBI** (Rapid Shallow Breathing Index) | RR / (TV in L) — gold standard weaning predictor; ≥105 indicates high failure risk |
| **P/F Ratio** | PaO₂ / FiO₂ — oxygenation index; <300 indicates ARDS spectrum |
| **PEEP** | Positive End-Expiratory Pressure — higher values suggest patient still needs ventilator support |
| **SpO₂** | Peripheral oxygen saturation |
| **Respiratory Rate** | Elevated RR is one of the strongest predictors of failure |
| **Arterial pH** | Acid-base status reflecting respiratory and metabolic function |

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [PhysioNet MIMIC-IV v3.1](https://physionet.org/content/mimiciv/) |
| **Total Patients** | 24,305 ICU stays |
| **Failure Rate** | 8.2% (1,990 failures / 22,315 successes) |
| **Features** | 13 clinical variables + 2 engineered features + 13 missingness indicators |
| **Time Window** | 24 hours prior to extubation attempt |
| **Outcome** | Binary — re-intubation or death within 72h post-extubation |

### Feature List
```
Vital Signs:      avg_heart_rate, avg_resp_rate, avg_spo2, avg_map, avg_temp_c
Blood Gas:        avg_ph, avg_pao2, avg_paco2
Ventilator:       avg_fio2, avg_peep, avg_tidal_volume
Labs:             avg_hemoglobin, avg_wbc
Demographics:     age, gender, age_masked (HIPAA flag)
Engineered:       rsbi, pf_ratio
Missingness:      <feature>_was_missing (×13)
```

---

## Project Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — Data Extraction                              [Saad]      │
│  BigQuery SQL on MIMIC-IV → mimic_data.csv (24,305 rows)           │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — Data Cleaning & Feature Engineering          [Faisal]    │
│  • Clinically-grounded outlier capping (13 features)               │
│  • HIPAA age masking flag                                           │
│  • Missingness indicators (_was_missing × 13)                      │
│  • Train/Test split FIRST → no data leakage                        │
│  • MICE imputation (IterativeImputer + BayesianRidge)              │
│  • Feature engineering: RSBI, P/F Ratio                            │
│  • VIF multicollinearity check                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — Exploratory Data Analysis                    [Abbas]     │
│  • 7 publication-quality visualizations                             │
│  • Class imbalance analysis (8.2% failure rate)                    │
│  • Mann-Whitney U significance tests                               │
│  • RSBI clinical threshold analysis (≥105)                         │
│  • HIPAA-masked patient risk stratification                        │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — Machine Learning Modeling                    [Mohammed]  │
│  • 3 candidate models: XGBoost, Random Forest, Balanced RF         │
│  • Imbalance handling: BorderlineSMOTE + BalancedRandomForest      │
│  • 5-Fold Stratified CV | Primary metric: PR-AUC                   │
│  • RandomizedSearchCV (50 trials × 5-fold per model)               │
│  • OOF threshold tuning (no test data leakage)                     │
│  • Probability calibration (isotonic + Brier score)                │
└────────────────────────────┬────────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — Interpretability & Advanced Evaluation       [Nawaf]     │
│  • SHAP values — clinical feature importance                       │
│  • Model explainability for clinical trust                         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Results

### Model Comparison (Test Set — n=4,861)

| Model | PR-AUC ⭐ | ROC-AUC | Recall | Precision | F1 | Threshold |
|-------|-----------|---------|--------|-----------|----|-----------|
| **Balanced RF** 🏆 | **0.4821** | 0.8762 | **0.709** | 0.326 | 0.447 | 0.59 |
| XGBoost | 0.4734 | **0.8752** | 0.480 | **0.492** | **0.486** | 0.51 |
| Random Forest | 0.4417 | 0.8676 | 0.520 | 0.397 | 0.450 | 0.51 |

> **Why PR-AUC?** With only 8.2% positive cases (severe class imbalance), ROC-AUC is misleading. PR-AUC focuses on the minority class (failures) and is the clinically meaningful metric here.

### Probability Calibration (Best Model — Balanced RF)
| Metric | Before Calibration | After Isotonic Calibration |
|--------|--------------------|---------------------------|
| **Brier Score** | 0.1449 | **0.0557** ✅ |

### Top 5 Clinical Predictors (Feature Importance)
```
1. avg_resp_rate    → 10.79%  (Elevated RR is the strongest failure signal)
2. avg_peep         →  8.65%  (High PEEP = patient still vent-dependent)
3. avg_pao2         →  6.81%  (Arterial oxygenation)
4. age              →  6.54%  (Older patients at higher risk)
5. avg_paco2        →  5.86%  (CO₂ retention = inadequate ventilation)
```

---

## Visualizations

| Plot | Description |
|------|-------------|
| `plot1_class_distribution.png` | Target class imbalance (bar + pie) |
| `plot2_correlation_heatmap.png` | Full feature correlation matrix |
| `plot3_violin_boxplots.png` | Vital signs: Success vs Failure (Mann-Whitney U) |
| `plot4_rsbi_distribution.png` | RSBI KDE with clinical threshold line |
| `plot5_missing_values.png` | Missing data before/after MICE imputation |
| `plot6_multivariate.png` | RR vs TV scatter + RSBI iso-lines |
| `plot7_age_masked_analysis.png` | HIPAA-masked patients failure rate analysis |
| `plot_ml_evaluation_v2.png` | PR curve, ROC curve, confusion matrix, threshold sweep |
| `plot_cv_feature_importance.png` | CV stability + feature importance bar chart |
| `plot_calibration.png` | Reliability diagram before/after isotonic calibration |

---

## Project Structure

```
Predicting-Extubation-Failure/
│
├── 📓 Predicting_Extubation_Failure.ipynb       ← Main notebook (local)
├── 📓 Predicting_Extubation_Failure_Colab.ipynb ← Google Colab version
│
├── 📊 Data (not tracked by Git — requires MIMIC-IV access)
│   ├── mimic_data.csv                           ← Raw extracted data
│   ├── mimic_data_cleaned_v2.csv                ← Cleaned dataset
│   ├── mimic_train.csv                          ← Training split
│   └── mimic_test.csv                           ← Testing split
│
├── 🤖 Models
│   ├── best_model.pkl                           ← Best model (Balanced RF)
│   └── best_model_calibrated.pkl               ← Probability-calibrated model
│
├── 📈 Plots
│   ├── plot1_class_distribution.png
│   ├── plot2_correlation_heatmap.png
│   ├── plot3_violin_boxplots.png
│   ├── plot4_rsbi_distribution.png
│   ├── plot5_missing_values.png
│   ├── plot6_multivariate.png
│   ├── plot7_age_masked_analysis.png
│   ├── plot_ml_evaluation_v2.png
│   ├── plot_cv_feature_importance.png
│   └── plot_calibration.png
│
└── README.md
```

> ⚠️ **Data Access Note:** The MIMIC-IV dataset requires credentialed access via [PhysioNet](https://physionet.org/content/mimiciv/). The CSV files are excluded from this repository. To reproduce results, complete the MIMIC-IV data use agreement and run Phase 1 (SQL extraction) in the notebook.

---

## Setup & Usage

### Option A — Google Colab (Recommended)

1. Upload `Predicting_Extubation_Failure_Colab.ipynb` to [Google Colab](https://colab.research.google.com)
2. Place the data files in your Google Drive at `MyDrive/ML_Project/`
3. Run all cells sequentially

### Option B — Local Environment

**Requirements:**
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn joblib
```

**Run:**
```bash
jupyter notebook Predicting_Extubation_Failure.ipynb
```

### Dependencies Overview

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` / `numpy` | Latest | Data manipulation |
| `scikit-learn` | ≥1.0 | Modeling, imputation, evaluation |
| `xgboost` | ≥1.7 | Gradient boosting classifier |
| `imbalanced-learn` | ≥0.10 | SMOTE, BalancedRandomForest |
| `matplotlib` / `seaborn` | Latest | Visualization |
| `scipy` | Latest | Statistical tests |
| `joblib` | Latest | Model serialization |

---

## Team

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Saad** | Team Lead | Data Extraction (SQL · MIMIC-IV BigQuery), Final Integration, Report Writing |
| **Nawaf** | Model Interpreter | SHAP Values, Advanced Evaluation, Interpretability |
| **Faisal** | Data Engineer | Data Cleaning, Missing Value Imputation, Feature Engineering |
| **Abbas** | Data Analyst | Exploratory Data Analysis, Statistical Visualizations |
| **Mohammed** | ML Modeler | Model Building, Hyperparameter Tuning, Evaluation |

---

## Methodology Highlights

### Why These Design Choices?

| Decision | Rationale |
|----------|-----------|
| **MICE over Mean Imputation** | Preserves joint feature distributions; clinically critical for correlated vitals |
| **Clinical bounds over IQR** | ICU values like HR>150 are real, not errors — blind statistical capping loses signal |
| **BorderlineSMOTE over SMOTE** | Oversamples only minority cases near the decision boundary — more informative for rare failures |
| **PR-AUC over ROC-AUC** | ROC-AUC is misleading at 8.2% positive rate; PR-AUC directly measures minority class performance |
| **OOF threshold tuning** | Threshold chosen from out-of-fold training predictions only — eliminates test data leakage |
| **Probability calibration** | ICU physicians need trustworthy probabilities (p̂=0.30 should reflect ~30% real risk) |
| **Missingness indicators** | Missing blood gas data itself is clinically informative (sicker patients often have missing labs) |

---

## Ethical & Clinical Disclaimer

> This model is developed for **academic research purposes only**. It is not validated for clinical use and should not be used to make or influence real patient care decisions without proper clinical validation, regulatory approval, and physician oversight.
>
> MIMIC-IV data is used under a credentialed data use agreement. Patient data has been de-identified per HIPAA Safe Harbor standards.

---

<div align="center">

**Dataset:** [MIMIC-IV — PhysioNet](https://physionet.org/content/mimiciv/) &nbsp;|&nbsp;
**Framework:** Python · scikit-learn · XGBoost

</div>
