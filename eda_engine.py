import pandas as pd
import numpy as np
from scipy import stats

# ---------------- COLOR TRACKER ----------------
def init_color_map(df):
    return pd.DataFrame("", index=df.index, columns=df.columns)

def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

# ---------------- MISSING VALUES ----------------
def handle_missing(df, col, method, color_map):
    mask = df[col].isna()

    if method == "mean" and is_numeric(df[col]):
        df.loc[mask, col] = df[col].mean()

    elif method == "median" and is_numeric(df[col]):
        df.loc[mask, col] = df[col].median()

    elif method == "mode":
        mode_val = df[col].mode(dropna=True)
        if not mode_val.empty:
            df.loc[mask, col] = mode_val[0]

    elif method == "drop":
        df = df.dropna(subset=[col])
        color_map = color_map.loc[df.index]
        return df, color_map

    elif method == "ffill":
        df[col] = df[col].fillna(method="ffill")

    elif method == "bfill":
        df[col] = df[col].fillna(method="bfill")

    elif method == "constant":
        df[col] = df[col].fillna(0)

    color_map.loc[mask, col] = "background-color: yellow"
    return df, color_map

# ---------------- OUTLIERS ----------------
def handle_outliers(df, col, method, color_map):
    if not is_numeric(df[col]):
        return df, color_map

    series = df[col]

    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (series < lower) | (series > upper)

    elif method == "zscore":
        z = np.abs(stats.zscore(series, nan_policy="omit"))
        mask = z > 3

    elif method == "cap":
        lower = series.quantile(0.05)
        upper = series.quantile(0.95)
        mask = (series < lower) | (series > upper)
        df.loc[mask, col] = np.clip(series[mask], lower, upper)

    else:
        return df, color_map

    color_map.loc[mask, col] = "background-color: red"
    return df, color_map

# ---------------- SCALING ----------------
def scale_column(df, col, method, color_map):
    if not is_numeric(df[col]):
        return df, color_map

    if method == "minmax":
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    elif method == "standard":
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    elif method == "log":
        df[col] = np.log1p(df[col])

    color_map[col] = "background-color: lightgreen"
    return df, color_map

# ---------------- ENCODING ----------------
def encode_column(df, col, method, color_map):
    if method == "label":
        df[col] = df[col].astype("category").cat.codes

    elif method == "onehot":
        dummies = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    color_map[col] = "background-color: cyan"
    return df, color_map

# ---------------- DATETIME ----------------
def parse_datetime(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

# ---------------- FEATURE ENGINEERING ----------------
def extract_date_features(df, col):
    df[col] = pd.to_datetime(df[col], errors="coerce")
    df[col + "_year"] = df[col].dt.year
    df[col + "_month"] = df[col].dt.month
    df[col + "_day"] = df[col].dt.day
    return df


# ================= NEW FUNCTIONS =================

def outlier_report(df):
    report = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        report[col] = int(count)

    return report


def correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr().round(3)

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample

# ---------------- DROP COLUMN ----------------
def drop_column(df, col, color_map):
    df = df.drop(columns=[col])
    color_map = color_map.drop(columns=[col])
    return df, color_map

# ---------------- TEXT ----------------
def handle_text(df, col, method, color_map):
    if method == "lowercase":
        df[col] = df[col].astype(str).str.lower()

    elif method == "remove_punctuation":
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r"[^\w\s]", "", x))

    elif method == "tfidf":
        tfidf = TfidfVectorizer(max_features=10)
        mat = tfidf.fit_transform(df[col].astype(str))
        tfidf_df = pd.DataFrame(mat.toarray(), columns=[f"{col}_tfidf_{i}" for i in range(mat.shape[1])])
        df = pd.concat([df.drop(columns=[col]), tfidf_df], axis=1)

    color_map[col] = "background-color: lightblue"
    return df, color_map

# ---------------- TYPE CAST ----------------
def change_type(df, col, method, color_map):
    if method == "to_numeric":
        df[col] = pd.to_numeric(df[col], errors="coerce")

    elif method == "to_category":
        df[col] = df[col].astype("category")

    color_map[col] = "background-color: orange"
    return df, color_map

# ---------------- FEATURE SELECTION ----------------
def select_features(df, col, method, color_map):
    if method == "variance":
        numeric = df.select_dtypes(include=np.number)
        low_var_cols = numeric.var()[numeric.var() < 0.01].index
        df = df.drop(columns=low_var_cols)
        color_map = color_map.drop(columns=low_var_cols)

    elif method == "correlation":
        corr = df.select_dtypes(include=np.number).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [c for c in upper.columns if any(upper[c] > 0.9)]
        df = df.drop(columns=drop_cols)
        color_map = color_map.drop(columns=drop_cols)

    return df, color_map

# ---------------- BALANCE ----------------
def balance_classes(df, col, method, color_map):
    counts = df[col].value_counts()
    max_count = counts.max()

    frames = []
    for cls in counts.index:
        subset = df[df[col] == cls]
        if method == "oversample":
            subset = resample(subset, replace=True, n_samples=max_count, random_state=42)
        else:
            subset = subset.sample(n=min(len(subset), max_count), random_state=42)
        frames.append(subset)

    df = pd.concat(frames)
    color_map = init_color_map(df)
    return df, color_map


# ---------------- MASTER FUNCTION ----------------
def apply_user_operations(df, operations):
    df = df.copy()
    color_map = init_color_map(df)

    for op in operations:
        t = op["type"]
        col = op.get("col")
        method = op.get("method")

        if col not in df.columns:
            continue

        if t == "missing":
            df, color_map = handle_missing(df, col, method, color_map)

        elif t == "outlier":
            df, color_map = handle_outliers(df, col, method, color_map)

        elif t == "scale":
            df, color_map = scale_column(df, col, method, color_map)

        elif t == "encode":
            df, color_map = encode_column(df, col, method, color_map)

        elif t == "datetime":
            df = parse_datetime(df, col)

        elif t == "feature":
            df = extract_date_features(df, col)

        elif t == "drop":
            df, color_map = drop_column(df, col, color_map)

        elif t == "text":
            df, color_map = handle_text(df, col, method, color_map)

        elif t == "type":
            df, color_map = change_type(df, col, method, color_map)

        elif t == "select":
            df, color_map = select_features(df, col, method, color_map)

        elif t == "balance":
            df, color_map = balance_classes(df, col, method, color_map)


    return df, color_map