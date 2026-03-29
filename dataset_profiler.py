import pandas as pd
import numpy as np


def detect_column_types(df):
    types = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            types[col] = "datetime"
        else:
            types[col] = "categorical"
    return types


def detect_target_column(df):
    candidates = []

    for col in df.columns:
        # Skip ID-like columns
        if any(x in col.lower() for x in ["id", "index", "no"]):
            continue

        nunique = df[col].nunique()
        ratio = nunique / len(df)

        # Skip useless columns
        if nunique <= 1:
            continue
        if ratio > 0.9:  # too unique → probably identifier
            continue

        score = 0

        name = col.lower()

        # Name hints
        if any(x in name for x in ["target", "label", "class", "outcome", "result", "status"]):
            score += 5
        if any(x in name for x in ["price", "salary", "score", "final", "grade", "g3"]):
            score += 3

        # Type hints
        if df[col].dtype == "object":
            score += 2
        if pd.api.types.is_numeric_dtype(df[col]):
            score += 1

        # Cardinality hint
        if 0.02 < ratio < 0.5:
            score += 3

        candidates.append((col, score))

    if not candidates:
        return None

    # Pick highest scoring column
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    return candidates[0][0]



def numeric_summary(df):
    summary = {}
    for col in df.select_dtypes(include="number"):
        summary[col] = {
            "mean": round(df[col].mean(), 3),
            "std": round(df[col].std(), 3),
            "min": round(df[col].min(), 3),
            "max": round(df[col].max(), 3),
            "skew": round(df[col].skew(), 3)
        }
    return summary


def categorical_summary(df):
    summary = {}
    for col in df.select_dtypes(exclude="number"):
        summary[col] = {
            "unique_values": int(df[col].nunique()),
            "top_value": df[col].mode()[0] if not df[col].mode().empty else None
        }
    return summary


def detect_outliers(df):
    outliers = {}
    for col in df.select_dtypes(include="number"):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        outliers[col] = int(count)
    return outliers


def profile_dataset(df):
    profile = {}
    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]
    profile["column_types"] = detect_column_types(df)

    profile["missing_values"] = (df.isnull().mean() * 100).round(2).to_dict()
    profile["duplicate_rows"] = int(df.duplicated().sum())

    profile["target_column"] = detect_target_column(df)
    profile["numeric_summary"] = numeric_summary(df)
    profile["categorical_summary"] = categorical_summary(df)
    profile["outliers"] = detect_outliers(df)

    return profile
