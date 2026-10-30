"""Train and compare grouped-validation classifiers for the source Zone_Match flag.

This target is a source-dataset label, not an observed crop yield or harvest result.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "master_dataset.csv"
METRICS_JSON = ROOT / "models" / "zone_match_metrics.json"
MODEL_FILE = ROOT / "models" / "zone_match_classifier.joblib"
SEED = 2026
CLASSES = ["No", "Yes"]

NUMERIC = [
    "pH_Mean", "EC_dSm", "OC_Pct", "N_kg_ac", "P_kg_ac", "K_kg_ac",
    "S_Sufficient_Pct", "Zn_Sufficient_Pct", "Fe_Sufficient_Pct",
    "Cu_Sufficient_Pct", "Mn_Sufficient_Pct", "B_Sufficient_Pct",
    "Mean_Annual_Rainfall_mm", "SW_Monsoon_Jun_Sep_mm", "NE_Monsoon_Oct_Dec_mm",
    "Soil_pH_Min", "Soil_pH_Max", "Water_Min_mm", "Water_Max_mm",
    "Duration_Min_Days", "Duration_Max_Days", "Temp_Min_C", "Temp_Max_C",
    "Humidity_Min_Pct", "Humidity_Max_Pct",
]
CATEGORICAL = [
    "Agro_Climatic_Zone", "Major_Soil_Type", "Crop_Name", "Crop_Category",
    "Required_Soil_Type", "Water_Source", "Primary_TN_Seasons",
]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "Zone_Match"


def make_pipeline(model, *, dense=False, scale_numeric=False):
    numeric_steps = [("fill", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)
    category_pipe = Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC),
        ("category", category_pipe, CATEGORICAL),
    ])
    return Pipeline([("prepare", prep), ("model", model)])


def score_predictions(y, predictions):
    return {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, predictions)), 4),
        "macro_f1": round(float(f1_score(y, predictions, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y, predictions, labels=CLASSES).tolist(),
        "classification_report": classification_report(y, predictions, labels=CLASSES, target_names=CLASSES, output_dict=True, zero_division=0),
    }


def main():
    data = pd.read_csv(SOURCE)
    data[TARGET] = data[TARGET].astype(str).str.strip().str.title()
    data = data[data[TARGET].isin(CLASSES)].copy()
    missing_features = sorted(set(FEATURES) - set(data.columns))
    if missing_features:
        raise ValueError(f"Source data is missing model fields: {missing_features}")

    X, y, groups = data[FEATURES], data[TARGET], data["District"]
    folds = GroupKFold(n_splits=5)
    model_factories = {
        "Most-frequent baseline": lambda: make_pipeline(DummyClassifier(strategy="most_frequent")),
        "Logistic Regression": lambda: make_pipeline(LogisticRegression(max_iter=1500, class_weight="balanced", C=0.5), scale_numeric=True),
        "Decision Tree": lambda: make_pipeline(DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, class_weight="balanced", random_state=SEED)),
        "Random Forest": lambda: make_pipeline(RandomForestClassifier(n_estimators=400, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)),
        "Extra Trees": lambda: make_pipeline(ExtraTreesClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        "Histogram Gradient Boosting": lambda: make_pipeline(HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=15, l2_regularization=2, random_state=SEED), dense=True),
    }

    results = {}
    for name, factory in model_factories.items():
        out_of_fold = np.empty(len(y), dtype=object)
        fold_scores = []
        for train_idx, test_idx in folds.split(X, y, groups=groups):
            pipeline = factory()
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            out_of_fold[test_idx] = pred
            fold_scores.append({
                "accuracy": float(accuracy_score(y.iloc[test_idx], pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y.iloc[test_idx], pred)),
                "macro_f1": float(f1_score(y.iloc[test_idx], pred, average="macro", zero_division=0)),
            })
        result = score_predictions(y, out_of_fold)
        result["fold_mean"] = {key: round(float(np.mean([f[key] for f in fold_scores]),), 4) for key in fold_scores[0]}
        result["fold_std"] = {key: round(float(np.std([f[key] for f in fold_scores], ddof=1)), 4) for key in fold_scores[0]}
        results[name] = result

    best_name = max(results, key=lambda name: results[name]["macro_f1"])
    final_model = model_factories[best_name]()
    final_model.fit(X, y)
    joblib.dump(final_model, MODEL_FILE, compress=3)

    # Permutation importance is measured on unseen districts in each fold.
    importance_by_fold = []
    for train_idx, test_idx in folds.split(X, y, groups=groups):
        fold_model = model_factories[best_name]()
        fold_model.fit(X.iloc[train_idx], y.iloc[train_idx])
        perm = permutation_importance(
            fold_model, X.iloc[test_idx], y.iloc[test_idx], scoring="f1_macro",
            n_repeats=4, random_state=SEED, n_jobs=1,
        )
        importance_by_fold.append(perm.importances_mean)
    means = np.mean(importance_by_fold, axis=0)
    top_features = sorted(
        [{"feature": name, "importance": round(float(value), 5)} for name, value in zip(FEATURES, means)],
        key=lambda item: item["importance"], reverse=True,
    )[:12]

    metrics = {
        "target": TARGET,
        "classes": CLASSES,
        "target_counts": {k: int(v) for k, v in y.value_counts().reindex(CLASSES, fill_value=0).items()},
        "rows": int(len(data)),
        "districts": int(data["District"].nunique()),
        "crops": int(data["Crop_Name"].nunique()),
        "validation": "Five-fold GroupKFold by District. Every validation fold contains districts excluded from that fold's training data.",
        "folds": 5,
        "features": FEATURES,
        "excluded_fields": ["District", "District_Named_In_Crop_Zones", "Key_Agro_Climatic_Zones", "Zone_Match"],
        "best_model": best_name,
        "results": results,
        "feature_importance": top_features,
        "feature_importance_method": "Mean permutation importance on held-out district folds, measured as macro-F1 change when a feature is shuffled.",
        "interpretation": "This model predicts the dataset's existing Yes/No Zone_Match flag. It does not predict observed crop success or yield.",
        "seed": SEED,
    }
    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Source rows: {len(data)}; districts: {data['District'].nunique()}; crops: {data['Crop_Name'].nunique()}")
    print("Source Zone_Match counts:", metrics["target_counts"])
    for name, result in results.items():
        print(f"{name}: macro_f1={result['macro_f1']:.3f}, balanced_accuracy={result['balanced_accuracy']:.3f}, accuracy={result['accuracy']:.3f}")
    print(f"Selected model by grouped macro-F1: {best_name}")
    print("Saved grouped-validation metrics and final classifier.")


if __name__ == "__main__":
    main()
