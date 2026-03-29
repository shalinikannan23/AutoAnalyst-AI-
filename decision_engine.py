import numpy as np
import pandas as pd


def decide_task_type(df, target):
    if pd.api.types.is_numeric_dtype(df[target]):
        if df[target].nunique() <= 20:
            return "classification"
        else:
            return "regression"
    else:
        return "classification"


def analyze_dataset_complexity(df):
    info = {}
    info["rows"] = df.shape[0]
    info["columns"] = df.shape[1]
    info["numeric_cols"] = len(df.select_dtypes(include="number").columns)
    info["categorical_cols"] = len(df.select_dtypes(exclude="number").columns)
    info["high_cardinality"] = [
        col for col in df.select_dtypes(exclude="number")
        if df[col].nunique() > 0.3 * len(df)
    ]
    return info


def preprocessing_decisions(missing_report, outlier_report, skew_report=None):
    decisions = []

    for col, miss in missing_report.items():
        if miss > 50:
            decisions.append(f"Drop column {col} (too many missing values)")
        elif miss > 20:
            decisions.append(f"Advanced imputation for {col} (KNN or model-based)")
        elif miss > 0:
            decisions.append(f"Simple imputation for {col} (mean/mode)")

    for col, out in outlier_report.items():
        if out > 0:
            decisions.append(f"Cap or remove outliers in {col}")

    if skew_report:
        for col, skew in skew_report.items():
            if abs(skew) > 1:
                decisions.append(f"Apply log/boxcox transform on {col}")

    return decisions


def select_models(task_type, dataset_info):
    models = []

    if task_type == "classification":
        models.append("LogisticRegression")

        if dataset_info["rows"] > 500:
            models.append("RandomForest")

        if dataset_info["categorical_cols"] == 0:
            models.append("XGBoost")

        if dataset_info["rows"] < 1000:
            models.append("KNN")

    else:  # regression
        models.append("LinearRegression")

        if dataset_info["rows"] > 500:
            models.append("RandomForestRegressor")

        if dataset_info["numeric_cols"] > 5:
            models.append("XGBoostRegressor")

    return list(set(models))


def detect_class_imbalance(df, target):
    value_counts = df[target].value_counts(normalize=True)
    if value_counts.max() > 0.75:
        return "Severe imbalance detected"
    elif value_counts.max() > 0.6:
        return "Moderate imbalance detected"
    return "No major imbalance"


def generate_decision_report(df, target, missing_report, outlier_report, skew_report):
    task = decide_task_type(df, target)
    dataset_info = analyze_dataset_complexity(df)
    imbalance = detect_class_imbalance(df, target) if task == "classification" else "Not applicable"

    preprocessing = preprocessing_decisions(missing_report, outlier_report, skew_report)
    models = select_models(task, dataset_info)

    return {
        "task_type": task,
        "dataset_complexity": dataset_info,
        "class_balance": imbalance,
        "preprocessing_steps": preprocessing,
        "recommended_models": models
    }

def get_imputation_options(df, missing_report):
    options = {}

    for col, miss in missing_report.items():
        if miss > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                options[col] = ["mean", "median", "mode"]
            else:
                options[col] = ["mode"]

    return options
