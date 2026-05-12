"""
Feature-selection ablation (Section 6.1).
SHAP-based RFE: trains XGBoost with Top-{53, 26, 13, 6, 3} features and
reports ROC-AUC, PR-AUC, F1, Recall, Precision on the held-out test set.

Run:    python3 run_feature_selection.py
Output: phase5_outputs/table_6_1_feature_selection.csv
        + console table
"""

import joblib
import pandas as pd
import time
import warnings

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, recall_score, precision_score,
)
from sklearn.preprocessing import RobustScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def main() -> None:
    X_tr = joblib.load("X_train.pkl")
    y_tr = joblib.load("y_train.pkl")
    X_te = joblib.load("X_test.pkl")
    y_te = joblib.load("y_test.pkl")
    print(f"Train: {X_tr.shape}   Test: {X_te.shape}")

    shap_df = pd.read_csv("phase5_outputs/table_5_5_shap_importance.csv")
    ranked = (
        shap_df.sort_values("Mean |SHAP|", ascending=False)["Feature"].tolist()
    )
    ranked = [f for f in ranked if f in X_tr.columns]
    print(f"SHAP-ranked features: {len(ranked)}")

    best_params = dict(
        learning_rate=0.03,
        max_depth=8,
        n_estimators=500,
        min_child_weight=7,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0,
        reg_lambda=5.0,
        gamma=0.3,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    THRESHOLD = 0.34
    results = []
    for k in [53, 26, 13, 6, 3]:
        feats = list(X_tr.columns) if k >= len(ranked) else ranked[:k]
        t0 = time.time()
        pipe = ImbPipeline([
            ("scaler", RobustScaler()),
            ("smote", BorderlineSMOTE(random_state=42)),
            ("clf", XGBClassifier(**best_params)),
        ])
        pipe.fit(X_tr[feats], y_tr)
        p = pipe.predict_proba(X_te[feats])[:, 1]
        pred = (p >= THRESHOLD).astype(int)
        row = dict(
            K=k,
            n_features=len(feats),
            ROC_AUC=round(roc_auc_score(y_te, p), 4),
            PR_AUC=round(average_precision_score(y_te, p), 4),
            F1=round(f1_score(y_te, pred), 4),
            Recall=round(recall_score(y_te, pred), 4),
            Precision=round(precision_score(y_te, pred), 4),
            time_s=round(time.time() - t0, 1),
        )
        results.append(row)
        print(row)

    out = pd.DataFrame(results)
    out_path = "phase5_outputs/table_6_1_feature_selection.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}\n")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
