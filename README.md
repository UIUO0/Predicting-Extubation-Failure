<div align="center">

# Predicting 72-Hour Extubation Failure in the ICU
### A TRIPOD-AI–Aligned Machine-Learning Pipeline on MIMIC-IV

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-0099CC)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.45%2B-8A2BE2)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-MIMIC--IV%20v3.1-C0392B)](https://physionet.org/content/mimiciv/3.1/)
[![Reporting](https://img.shields.io/badge/Reporting-TRIPOD--AI-1D9E75)](https://www.tripod-statement.org/)
[![License](https://img.shields.io/badge/License-Academic%20%2F%20Research-blue)]()

</div>

> **Summary.** A reproducible, leakage-safe ML pipeline that predicts the probability of **re-intubation or death within 72 h of planned extubation** using 24 h of pre-extubation ICU data (vitals, blood gas, ventilator settings, labs, GCS). Three tree-ensembles are benchmarked with full discrimination, calibration, statistical testing, interpretability (SHAP), risk stratification, and subgroup-fairness auditing.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Scientific Contributions](#2-scientific-contributions)
3. [Cohort & Outcome](#3-cohort--outcome)
4. [Methodological Pipeline](#4-methodological-pipeline)
5. [Headline Results](#5-headline-results)
6. [Reproducibility](#6-reproducibility)
7. [Repository Structure](#7-repository-structure)
8. [Installation & Execution](#8-installation--execution)
9. [TRIPOD-AI Cross-Reference](#9-tripod-ai-cross-reference)
10. [Limitations](#10-limitations)
11. [Ethics & Data Governance](#11-ethics--data-governance)
12. [References](#12-references)
13. [Team](#13-team)

---

## 1. Motivation

Mechanical ventilation is life-sustaining, yet **premature extubation** followed by re-intubation within 72 h is an independent predictor of in-hospital mortality (OR 2.5–5.0; +12 days ICU stay; Esteban 2013). The bedside gold standard — the Rapid Shallow Breathing Index (RSBI ≥ 105; Yang & Tobin 1991) — rarely exceeds AUROC 0.70 and ignores multimodal physiology.

> **Research question.** Can a tree-ensemble classifier trained on 24 h of pre-extubation ICU data produce **calibrated, explainable, subgroup-robust** probability estimates of 72-h extubation failure?

---

## 2. Scientific Contributions

| # | Contribution | Where |
|---|--------------|-------|
| C1 | **Leakage-safe split** — subject-level `GroupShuffleSplit` (0 overlapping `subject_id`) | §Phase 2.5 |
| C2 | **Clinically-grounded cleaning** — physiological bounds (21 vars) + MICE (train-only fit) + post-imputation re-clipping | §Phase 2.4–2.6 |
| C3 | **Principled outcome** — re-intubation anchored to new ETT `procedureevents` (1–72 h window), not "any new ICU admission" | §Phase 1 SQL |
| C4 | **Rigorous evaluation** — OOF threshold tuning, isotonic calibration, DeLong / McNemar / permutation tests, BCa-bootstrap CIs (N = 1000), Decision Curve Analysis | §Phase 5.1–5.4 |
| C5 | **Multi-level interpretability** — TreeSHAP global (beeswarm, bar, dependence) + local waterfalls for TP/FN/FP/TN archetypes | §Phase 5.5–5.6 |
| C6 | **Operational risk stratification** — 4 clinician-facing tiers with PPV and **Number Needed to Evaluate (NNE)** | §Phase 5.7 |
| C7 | **Fairness audit** — per-subgroup ROC-AUC, sensitivity, specificity across age, GCS, vent-duration, comorbidity, gender | §Phase 5.8 |
| C8 | **TRIPOD-AI–aligned reporting** — every figure/table persisted with `manifest.csv` + machine-readable `summary.json` | §Phase 5.9 |

---

## 3. Cohort & Outcome

### 3.1 Cohort

| Property | Value |
|----------|-------|
| **Source** | [PhysioNet MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) (Johnson 2023) |
| **Extraction** | BigQuery — `physionet-data.mimiciv_3_1_{icu,hosp}` |
| **Inclusion** | Adult ICU stays with ≥ 1 documented extubation `procedureevent` (itemids 227194 / 225468 / 225477); first extubation per stay |
| **N total** | **24,305** ICU stays |
| **N train / test** | 19,445 / 4,860 — subject-disjoint (`GroupShuffleSplit`, seed = 7 chosen from 20 candidates for prevalence match) |
| **Prevalence** | 24.45 % (test) / 24.48 % (train) |
| **Features** | 29 raw → **53** after engineering + missingness channels |
| **Window** | 24 h pre-extubation |

### 3.2 Outcome

`failure_label = 1` if **either** event occurs in `(extubation_time + 1h, extubation_time + 72h]`:

1. **Re-intubation** — new ETT-insertion event (`itemid ∈ {224263, 224264, 224267}`) or invasive vent event (`225792`).
2. **Death** — `admissions.deathtime` in `[extubation_time, extubation_time + 72h]`.

The 1-h lower bound excludes peri-procedural airway manipulation; the 72-h horizon is the standard clinical attribution window (Esteban 2013; Thille 2011).

### 3.3 Feature Families

| Family | Variables |
|--------|-----------|
| **Vitals (central)** | `avg_heart_rate`, `avg_resp_rate`, `avg_spo2`, `avg_map`, `avg_temp_c` |
| **Vitals (variability)** | `std_heart_rate`, `std_resp_rate`, `max_resp_rate`, `min_spo2` |
| **Blood gas** | `avg_ph`, `avg_pao2`, `avg_paco2` |
| **Ventilator** | `avg_fio2`, `avg_peep`, `avg_tidal_volume`, `vent_duration_hours` |
| **Labs** | `avg_hemoglobin`, `avg_wbc` |
| **Neurological** | `avg_gcs_eye`, `avg_gcs_verbal`, `avg_gcs_motor`, `gcs_total` (engineered) |
| **Pharm. / burden** | `on_vasopressor`, `comorbidity_count` |
| **Demographics** | `age`, `gender`, `age_masked` (HIPAA ≥ 89 flag) |
| **Engineered indices** | `rsbi = RR / TV(L)`, `pf_ratio = PaO₂ / FiO₂`, `gcs_total` |
| **Missing-indicator channel** | `<feature>_was_missing` (×21) — encodes informative missingness |

---

## 4. Methodological Pipeline

```
Phase 1  Data Extraction         BigQuery SQL → mimic_data.csv [24,305 × 29]
Phase 2  Cleaning & Engineering  Bounds → GroupSplit → Missing flags → MICE → RSBI / P/F / GCS_total → VIF
Phase 3  Exploratory Analysis    Imbalance, correlation, violin + Mann–Whitney, RSBI KDE, RR–TV multivariate
Phase 4  Modeling & Calibration  XGB / RF / Balanced-RF · StratifiedGroupKFold · RandomizedSearchCV (50×5) · OOF threshold · Isotonic
Phase 5  Advanced Evaluation     §5.1 Discrimination · §5.2 Calibration · §5.3 DeLong/McNemar/Permutation
                                 §5.4 Bootstrap + DCA · §5.5–5.6 SHAP · §5.7 Risk strata · §5.8 Fairness · §5.9 Manifest
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **`GroupShuffleSplit` on `subject_id`** | Multiple ICU stays per patient → row-wise splits leak; group-splitting is TRIPOD-AI compliant. |
| **Physiological bounds before statistical capping** | HR > 150 or SpO₂ < 80 are clinically real in ICU; blind IQR capping destroys signal. |
| **MICE over median imputation** | Preserves inter-feature dependencies needed by derived indices (RSBI, P/F). Fit-on-train ⇒ no leakage. |
| **Missing-indicators channel** | Missingness in MIMIC is non-random and clinically informative. |
| **BorderlineSMOTE** | Oversamples minority points near the decision boundary; avoids synthesising implausible failures. |
| **PR-AUC as primary metric** | At 24 % prevalence, PR-AUC is more sensitive to minority recall than ROC-AUC. |
| **OOF threshold tuning** | Operating threshold chosen only on OOF train predictions; the test set is touched **once**. |
| **Isotonic calibration + Brier** | Non-parametric, consistently outperforms Platt on tree ensembles. |
| **DCA over 5–30 % thresholds** | Encodes asymmetric cost: a missed failure is far worse than a false alarm. |

---

## 5. Headline Results

> Best model selected by **5-fold group-CV PR-AUC: XGBoost**. Test set is held out and touched **once**.

### 5.1 Cross-Validation (5-fold StratifiedGroupKFold)

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|-------|:-------:|:------:|:------:|:---------:|:--:|
| **XGBoost** 🏆 | 0.749 ± 0.005 | **0.534 ± 0.015** | 0.351 ± 0.004 | 0.611 ± 0.021 | 0.446 ± 0.005 |
| Random Forest | 0.737 ± 0.008 | 0.496 ± 0.019 | 0.512 ± 0.007 | 0.466 ± 0.016 | 0.488 ± 0.011 |
| Balanced RF | 0.745 ± 0.009 | 0.529 ± 0.015 | 0.700 ± 0.013 | 0.389 ± 0.014 | 0.500 ± 0.014 |

### 5.2 Held-Out Test (n = 4,860, prevalence = 24.4 %)

| Model | Thr. | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|-------|:----:|:-------:|:------:|:--:|:------:|:---------:|
| **XGBoost** 🏆 | 0.34 | **0.757** | **0.545** | **0.525** | 0.588 | 0.474 |
| Random Forest | 0.40 | 0.749 | 0.533 | 0.502 | 0.614 | 0.425 |
| Balanced RF | 0.54 | 0.746 | 0.530 | 0.499 | **0.645** | 0.407 |

Phase 5.4 reports BCa-bootstrap 95 % CIs (N = 1000) for every metric.

### 5.3 Calibration

| | Pre-calibration | Post-isotonic |
|---|:-:|:-:|
| **Brier score** | 0.1527 | **0.1489** ✅ |

Phase 5.2 additionally reports ECE, Brier decomposition (Murphy 1973: Reliability − Resolution + Uncertainty), and the Hosmer–Lemeshow GoF test.

### 5.4 Top Gini Predictors (XGBoost)

1. `on_vasopressor` — 0.128
2. `gender` — 0.114
3. `min_spo2` — 0.082
4. `vent_duration_hours_was_missing` — 0.063
5. `comorbidity_count` — 0.042

> Directional / non-linear interpretation is provided by the SHAP analysis in Phase 5.5–5.6.

---

## 6. Reproducibility

| Control | Implementation |
|---------|----------------|
| **Global seed** | `RANDOM_STATE = 42` for NumPy, scikit-learn, XGBoost, SMOTE, bootstrap. |
| **Split seed** | `GroupShuffleSplit` searched over seeds 0–19 → seed 7 (prevalence Δ = 0.03 pp). |
| **Bootstrap** | Per-iteration seeding for exact replay (N = 1000, stratified, BCa). |
| **Phase isolation** | Phase 5 runs standalone by loading `best_model.pkl`, `X_test.pkl`, `y_test.pkl`, `results.pkl`, `best_name.pkl`. |
| **Artifact manifest** | All Phase 5 outputs indexed in `phase5_outputs/manifest.csv` + `summary.json`. |

---

## 7. Repository Structure

```
Predicting-Extubation-Failure/
├── Predicting_Extubation_Failure.ipynb   ← Master notebook (Phases 1–5)
├── README.md                             ← This document
├── .gitignore                            ← Excludes *.csv & *.pkl (PHI + large)
│
├── Data (git-ignored; credentialed access required)
│   ├── mimic_data.csv                    ← Raw extraction
│   ├── mimic_data_cleaned_v2.csv         ← Cleaned + engineered
│   ├── mimic_train.csv  /  mimic_test.csv
│
├── Persisted artifacts (git-ignored)
│   ├── best_model.pkl                    ← Winning XGBoost Pipeline
│   ├── best_model_calibrated.pkl         ← Isotonic wrapper
│   ├── X_test.pkl  /  y_test.pkl
│
├── EDA figures (Phase 3)                 ← plot1 … plot7 *.png
├── Phase 4 figures                       ← plot_ml_evaluation_v2 / cv_feature_importance / calibration
│
└── phase5_outputs/                       ← Auto-generated
    ├── table_5_1_discrimination.csv     · figure_5_1_discrimination.png
    ├── table_5_2_calibration.csv        · figure_5_2_calibration.png
    ├── table_5_3_statistical_comparison.csv
    ├── table_5_4a_bootstrap_ci.csv      · table_5_4b_dca_anchors.csv
    ├── figure_5_4_decision_curve.png
    ├── table_5_5_shap_importance.csv    · figure_5_5_shap_global.png  · figure_5_5b_shap_dependence.png
    ├── figure_5_6_waterfall_*.png       (TP / FN / FP / TN)
    ├── table_5_7_risk_stratification.csv · figure_5_7_risk_strata.png
    ├── table_5_8_subgroup_audit.csv     · figure_5_8_subgroup_audit.png
    ├── manifest.csv                     · summary.json
```

---

## 8. Installation & Execution

### 8.1 Requirements
- Python ≥ 3.10, ≥ 8 GB RAM. Phase 4 tuning ≈ 60 min on a modern laptop; other phases < 5 min each.

### 8.2 Environment

```bash
git clone https://github.com/UIUO0/Predicting-Extubation-Failure.git
cd Predicting-Extubation-Failure

python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "pandas>=2.0" "numpy>=1.24" "scipy>=1.11" \
            "scikit-learn>=1.4" "xgboost>=2.0" \
            "imbalanced-learn>=0.12" "shap>=0.45" \
            "matplotlib>=3.7" "seaborn>=0.13" \
            "statsmodels>=0.14" "joblib>=1.3" jupyter
```

### 8.3 Data Access (credentialed)
1. Complete CITI "Data or Specimens Only Research" training.
2. Sign the MIMIC-IV DUA at [PhysioNet](https://physionet.org/content/mimiciv/3.1/).
3. Authenticate to BigQuery and run the Phase 1 SQL block.

### 8.4 Execute
```bash
jupyter notebook Predicting_Extubation_Failure.ipynb
```
Phase 5 can be re-run standalone — the §5.0 safe-load block auto-detects the persisted pickles.

---

## 9. TRIPOD-AI Cross-Reference

TRIPOD-AI (Collins 2024) supersedes TRIPOD (Collins 2015) for AI/ML clinical prediction models.

| TRIPOD-AI Item | Location |
|----------------|----------|
| 1 — Title & abstract | §1, §2 |
| 3 — Source of data | §3.1 |
| 5a–b — Eligibility & outcome | §3.1, §3.2 |
| 6 — Predictors | §3.3, §4 |
| 7 — Sample size | §3.1 |
| 8 — Missing data | §4 (MICE + indicators) |
| 10a–d — Model development | Phase 4 |
| 10e — Discrimination, calibration, clinical utility | §5 + Phase 5.1–5.4 |
| 11 — Risk groups | Phase 5.7 |
| 13 — Fairness / subgroup | Phase 5.8 |
| 17 — Interpretability | Phase 5.5–5.6 |
| 19 — Limitations | §10 |
| 20 — Data & code availability | §6, §7 |

---

## 10. Limitations

A predictive model is only as trustworthy as its disclosed weaknesses.

1. **Single-center fit, no external validation.** All metrics are internal to MIMIC-IV (BIDMC). Replication on **eICU-CRD / AmsterdamUMCdb / HiRID** is the immediate next step.
2. **No temporal validation.** The split is random across subjects, not across time. Protocol drift may inflate apparent performance.
3. **Outcome-definition sensitivity.** `re-intubation` depends on `procedureevents` fidelity; NIV rescue and comfort-care withdrawal can confound. No sensitivity analysis over the label definition is presented.
4. **Proxy comorbidity burden.** `comorbidity_count` is a raw distinct-ICD count, not a weighted Charlson / Elixhauser score.
5. **Associative, not causal.** SHAP attributions are **not** intervention targets. Counterfactual / do-calculus analyses (e.g., DoWhy, DiCE) are out of scope.
6. **Window rigidity.** A 24-h aggregation smooths short-lived but decisive events (e.g., 10-min desaturations). Sequence models over raw `chartevents` are a natural extension.
7. **RSBI baseline not benchmarked standalone.** RSBI enters as a feature only; head-to-head comparison vs. RSBI-alone is required for any clinical-impact claim.
8. **Limited fairness axes.** Race / ethnicity (available in MIMIC) is **not yet** stratified — required to complete the equity audit (Obermeyer 2019).
9. **Retrospective only.** No prospective or silent-mode deployment.

---

## 11. Ethics & Data Governance

- **IRB.** MIMIC-IV is approved by BIDMC & MIT IRBs under a waiver of informed consent (protocol 2001-P-001699/14).
- **De-identification.** All PHI is de-identified per HIPAA Safe Harbor; ages ≥ 89 are masked to 91 (preserved via the `age_masked` flag).
- **Distribution.** No CSV / pickle artifacts are redistributed in this repo (see `.gitignore`). Reproduction requires independent credentialed access.
- **Clinical disclaimer.** Research prototype only. Not a CE / FDA-cleared device. No prospective validation, human-factors testing, or regulatory review. **Must not** influence individual patient decisions.

---

## 12. References

**Data & cohort**
- Johnson AEW *et al.* MIMIC-IV, a freely accessible EHR dataset. *Scientific Data* 10:1 (2023). [doi:10.1038/s41597-022-01899-x](https://doi.org/10.1038/s41597-022-01899-x)
- Goldberger AL *et al.* PhysioBank / PhysioToolkit / PhysioNet. *Circulation* 101 (2000): e215–e220.

**Clinical context**
- Esteban A *et al.* Evolution of mortality over time in patients receiving mechanical ventilation. *Am J Respir Crit Care Med* 188 (2013): 220–230.
- Thille AW *et al.* Outcomes of extubation failure. *Crit Care Med* 39 (2011): 2612–2618.
- Yang KL, Tobin MJ. Indexes predicting weaning outcome (RSBI). *NEJM* 324 (1991): 1445–1450.

**Methodology**
- Collins GS *et al.* TRIPOD+AI statement. *BMJ* 385 (2024): e078378.
- Van Calster B *et al.* Calibration: the Achilles heel of predictive analytics. *BMC Medicine* 17 (2019): 230.
- DeLong ER *et al.* Comparing areas under correlated ROC curves. *Biometrics* 44 (1988): 837–845.
- Vickers AJ, Elkin EB. Decision curve analysis. *Med Decis Making* 26 (2006): 565–574.
- Lundberg SM *et al.* From local explanations to global understanding with explainable AI for trees. *Nat Mach Intell* 2 (2020): 56–67.
- Chawla NV *et al.* SMOTE. *JAIR* 16 (2002): 321–357. · Han, Wang & Mao. Borderline-SMOTE. *ICIC* (2005).
- Van Buuren S, Groothuis-Oudshoorn K. `mice`: MICE in R. *J Stat Softw* 45 (2011): 1–67.
- Steyerberg EW. *Clinical Prediction Models* (2nd ed.), Springer, 2019.
- Murphy AH. Probability score decomposition. *J Appl Meteorol* 12 (1973): 595–600.

**Fairness**
- Obermeyer Z *et al.* Dissecting racial bias in an algorithm. *Science* 366 (2019): 447–453.
- Pleiss G *et al.* On fairness and calibration. *NeurIPS* (2017).

---

## 13. Team

| Member | Role | Contribution |
|--------|------|--------------|
| **Saad** | Team Lead | Data extraction (BigQuery SQL on MIMIC-IV), integration, manuscript |
| **Faisal** | Data Engineer | Cleaning, physiological bounds, MICE imputation, feature engineering |
| **Abbas** | Data Analyst | Exploratory data analysis, statistical visualisations |
| **Mohammed** | ML Modeler | Pipelines, hyper-parameter search, threshold tuning, calibration |
| **Nawaf** | Model Interpreter | Advanced evaluation, SHAP, fairness audit, TRIPOD-AI reporting |

---

<div align="center">

**Dataset** [MIMIC-IV v3.1 · PhysioNet](https://physionet.org/content/mimiciv/3.1/) &nbsp;·&nbsp;
**Reporting** [TRIPOD-AI 2024](https://www.tripod-statement.org/) &nbsp;·&nbsp;
**Stack** Python · scikit-learn · XGBoost · imbalanced-learn · SHAP

*This is a research prototype. Not a medical device.*

</div>
