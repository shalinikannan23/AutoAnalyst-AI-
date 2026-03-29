import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

# ---------------- ENCODING ----------------
def encode_if_needed(df):
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == "object" or df_encoded[col].dtype.name == "category":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded


# ---------------- STABILITY SCORE ----------------
def stability_score(X, y, feature, rounds=20):
    scores = []
    for _ in range(rounds):
        sample = X.sample(frac=0.8, replace=True)
        scores.append(sample[feature].corr(y.loc[sample.index]))
    return np.std(scores)


# ---------------- FEATURE SCORING ----------------
def compute_feature_scores(df, target):
    df = df.copy()

    df_enc = encode_if_needed(df)

    X = df_enc.drop(columns=[target])
    y = df_enc[target]

    X = X.fillna(X.median(numeric_only=True))
    y = y.fillna(y.median())

    # Correlation
    corr = X.corrwith(y).fillna(0)

    # Direction
    direction = X.corrwith(y).fillna(0)

    # Mutual Information
    if y.nunique() <= 20:
        mi = mutual_info_classif(X, y, discrete_features="auto")
    else:
        mi = mutual_info_regression(X, y)

    mi = pd.Series(mi, index=X.columns)

    # Variance
    var = X.var()

    # Normalize
    def norm(series):
        return (series - series.min()) / (series.max() - series.min() + 1e-9)

    corr_n = norm(corr.abs())
    mi_n = norm(mi)
    var_n = norm(var)

    final_score = (corr_n + mi_n + var_n) / 3

    # Stability
    stability = {}
    for col in X.columns:
        stability[col] = stability_score(X, y, col)

    stability = pd.Series(stability)

    result = pd.DataFrame({
        "Correlation": corr.round(3),
        "Direction": direction.round(3),
        "Mutual_Information": mi.round(3),
        "Variance": var.round(3),
        "Stability": stability.round(4),
        "Final_Score": (final_score * 100).round(2)
    }).sort_values("Final_Score", ascending=False)

    # Explanation text
    def explain(row):
        if row["Direction"] > 0:
            trend = "increases"
        else:
            trend = "decreases"

        return f"{row.name} {trend} the target and has influence strength of {row['Final_Score']}%."

    result["Explanation"] = result.apply(explain, axis=1)

    return result


# ---------------- REDUNDANT FEATURES ----------------
def find_redundant_features(df, threshold=0.9):
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    redundant = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > threshold:
                redundant.append((corr_matrix.columns[i], corr_matrix.columns[j]))

    return redundant


# ---------------- DATASET INSIGHT ----------------
def generate_feature_insights(score_df, redundant_pairs):
    top = score_df.head(3).index.tolist()
    low = score_df.tail(3).index.tolist()

    text = []
    text.append(f"The most influential features are {', '.join(top)}.")
    text.append(f"The least useful features are {', '.join(low)}.")

    if redundant_pairs:
        pairs = [f"{a} & {b}" for a, b in redundant_pairs[:3]]
        text.append(f"Highly redundant feature pairs detected: {', '.join(pairs)}.")

    text.append("Scores are computed using correlation, mutual information, variance and stability.")
    text.append("Direction indicates whether the feature increases or decreases the target.")

    return " ".join(text)
