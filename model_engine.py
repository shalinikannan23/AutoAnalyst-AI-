import numpy as np
import pandas as pd


def analyze_feature_types(df, target=None):
    if target and target in df.columns:
        X = df.drop(columns=[target])
    else:
        X = df.copy()

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    return {
        "num_features": len(num_cols),
        "cat_features": len(cat_cols),
        "total_features": X.shape[1]
    }


# ---------------- CLASSIFICATION ----------------
def recommend_classification_models(feature_info, dataset_size):
    models = {}

    models["Logistic Regression"] = {
        "when_to_use": (
            "Baseline, interpretable classifier\n"
            "Binary classification problems\n"
            "Works well with linearly separable data"
        ),
        "important_params": {
            "C": "0.01 to 10 (regularization strength)",
            "solver": "liblinear, saga",
            "class_weight": "balanced if data is imbalanced",
            "max_iter": "100 to 500 iterations",
            "penalty": "l1, l2, elasticnet"
        }
    }

    models["KNN"] = {
        "when_to_use": (
            "Small datasets with distance-based patterns\n"
            "Non-parametric data\n"
            "Sensitive to irrelevant features"
        ),
        "important_params": {
            "n_neighbors": "3 to 15",
            "metric": "euclidean or manhattan",
            "weights": "uniform or distance",
            "algorithm": "auto, ball_tree, kd_tree"
        }
    }

    models["Naive Bayes"] = {
        "when_to_use": (
            "Text or probability-based features\n"
            "Works well with high dimensional data\n"
            "Assumes feature independence"
        ),
        "important_params": {
            "var_smoothing": "1e-9 to 1e-6",
            "priors": "None or array of class priors",
            "fit_prior": "True or False"
        }
    }

    models["Decision Tree"] = {
        "when_to_use": (
            "Rule-based and interpretable\n"
            "Handles both numerical and categorical data\n"
            "Sensitive to overfitting"
        ),
        "important_params": {
            "max_depth": "3 to 20",
            "min_samples_split": "2 to 10",
            "criterion": "gini or entropy",
            "min_samples_leaf": "1 to 5"
        }
    }

    models["Random Forest"] = {
        "when_to_use": (
            "Non-linear relationships\n"
            "Reduces overfitting compared to Decision Tree\n"
            "Works well with missing values"
        ),
        "important_params": {
            "n_estimators": "100 to 500",
            "max_depth": "5 to 30",
            "max_features": "sqrt, log2, None",
            "min_samples_split": "2 to 10"
        }
    }

    models["Gradient Boosting"] = {
        "when_to_use": (
            "Higher accuracy than RF\n"
            "Handles mixed types of features\n"
            "Sensitive to noisy data"
        ),
        "important_params": {
            "n_estimators": "100 to 300",
            "learning_rate": "0.01 to 0.3",
            "max_depth": "3 to 8",
            "subsample": "0.5 to 1.0",
            "min_samples_split": "2 to 10"
        }
    }

    if feature_info["num_features"] > 3:
        models["XGBoost"] = {
            "when_to_use": (
                "High performance structured data\n"
                "Handles missing values well\n"
                "Good for imbalanced datasets"
            ),
            "important_params": {
                "n_estimators": "100 to 300",
                "learning_rate": "0.01 to 0.3",
                "max_depth": "3 to 8",
                "subsample": "0.5 to 1.0",
                "colsample_bytree": "0.5 to 1.0"
            }
        }

    models["SVM"] = {
        "when_to_use": (
            "High dimensional data\n"
            "Effective for small to medium datasets\n"
            "Sensitive to feature scaling"
        ),
        "important_params": {
            "C": "0.1 to 10",
            "kernel": "linear or rbf",
            "gamma": "scale or auto",
            "degree": "for poly kernel"
        }
    }

    return models


# ---------------- REGRESSION ----------------
def recommend_regression_models(feature_info, dataset_size):
    models = {}

    models["Linear Regression"] = {
        "when_to_use": (
            "Linear relationships\n"
            "Baseline regression\n"
            "Sensitive to outliers"
        ),
        "important_params": {
            "fit_intercept": "True or False",
            "normalize": "True or False",
            "copy_X": "True or False"
        }
    }

    models["Ridge"] = {
        "when_to_use": (
            "Multicollinearity present\n"
            "Linear regularized regression\n"
            "Reduces overfitting"
        ),
        "important_params": {
            "alpha": "0.1 to 10",
            "fit_intercept": "True or False",
            "solver": "auto, svd, cholesky"
        }
    }

    models["Lasso"] = {
        "when_to_use": (
            "Feature selection needed\n"
            "Sparse model desired\n"
            "Reduces irrelevant features"
        ),
        "important_params": {
            "alpha": "0.01 to 1",
            "max_iter": "1000 to 5000",
            "tol": "1e-4 to 1e-2"
        }
    }

    models["ElasticNet"] = {
        "when_to_use": (
            "Combination of Ridge and Lasso\n"
            "Handles multicollinearity\n"
            "Balances sparsity and regularization"
        ),
        "important_params": {
            "alpha": "0.01 to 1",
            "l1_ratio": "0.1 to 0.9",
            "max_iter": "1000 to 5000",
            "tol": "1e-4 to 1e-2"
        }
    }

    models["KNN Regressor"] = {
        "when_to_use": (
            "Local patterns\n"
            "Non-linear regression\n"
            "Sensitive to feature scaling"
        ),
        "important_params": {
            "n_neighbors": "3 to 15",
            "weights": "uniform or distance",
            "metric": "euclidean or manhattan"
        }
    }

    models["Decision Tree Regressor"] = {
        "when_to_use": (
            "Rule-based regression\n"
            "Interpretable results\n"
            "Can overfit easily"
        ),
        "important_params": {
            "max_depth": "3 to 20",
            "min_samples_split": "2 to 10",
            "min_samples_leaf": "1 to 5"
        }
    }

    models["Random Forest Regressor"] = {
        "when_to_use": (
            "Non-linear regression\n"
            "Reduces overfitting\n"
            "Handles missing values"
        ),
        "important_params": {
            "n_estimators": "100 to 500",
            "max_depth": "5 to 30",
            "min_samples_split": "2 to 10",
            "max_features": "sqrt, log2, None"
        }
    }

    models["Gradient Boosting Regressor"] = {
        "when_to_use": (
            "High accuracy\n"
            "Handles mixed data types\n"
            "Sensitive to noisy data"
        ),
        "important_params": {
            "learning_rate": "0.01 to 0.3",
            "n_estimators": "100 to 300",
            "max_depth": "3 to 8",
            "subsample": "0.5 to 1.0",
            "min_samples_split": "2 to 10"
        }
    }

    if feature_info["num_features"] > 5:
        models["XGBoost Regressor"] = {
            "when_to_use": (
                "High accuracy structured regression\n"
                "Handles missing values\n"
                "Good for imbalanced data"
            ),
            "important_params": {
                "n_estimators": "100 to 300",
                "learning_rate": "0.01 to 0.3",
                "max_depth": "3 to 8",
                "subsample": "0.5 to 1.0",
                "colsample_bytree": "0.5 to 1.0"
            }
        }

    return models


# ---------------- CLUSTERING ----------------
def recommend_clustering_models(feature_info, dataset_size):
    models = {}

    models["KMeans"] = {
        "when_to_use": (
            "Well-separated spherical clusters\n"
            "Scales well for large datasets\n"
            "Sensitive to initialization"
        ),
        "important_params": {
            "n_clusters": "2 to 10",
            "init": "k-means++",
            "n_init": "10 to 50",
            "max_iter": "300 to 500"
        }
    }

    models["Hierarchical Clustering"] = {
        "when_to_use": (
            "Small datasets and dendrogram analysis\n"
            "Does not require number of clusters upfront\n"
            "Computationally expensive for large data"
        ),
        "important_params": {
            "linkage": "ward, complete, average",
            "affinity": "euclidean, manhattan"
        }
    }

    models["DBSCAN"] = {
        "when_to_use": (
            "Noise and irregular clusters\n"
            "Does not require number of clusters\n"
            "Good for arbitrary-shaped clusters"
        ),
        "important_params": {
            "eps": "0.2 to 2",
            "min_samples": "3 to 10",
            "metric": "euclidean, manhattan"
        }
    }

    models["Gaussian Mixture"] = {
        "when_to_use": (
            "Probabilistic clustering\n"
            "Soft assignments of points\n"
            "Handles overlapping clusters"
        ),
        "important_params": {
            "n_components": "2 to 10",
            "covariance_type": "full, tied",
            "max_iter": "100 to 500"
        }
    }

    models["Spectral Clustering"] = {
        "when_to_use": (
            "Complex cluster shapes\n"
            "Good for small datasets\n"
            "Requires similarity matrix"
        ),
        "important_params": {
            "n_clusters": "2 to 10",
            "affinity": "nearest_neighbors, rbf",
            "assign_labels": "kmeans or discretize"
        }
    }

    return models


# ---------------- STRATEGY ----------------
def generate_model_strategy(df, target, task_type):
    task_type = task_type.lower().strip()

    if task_type == "classification":
        feature_info = analyze_feature_types(df, target)
        models = recommend_classification_models(feature_info, len(df))

    elif task_type == "regression":
        feature_info = analyze_feature_types(df, target)
        models = recommend_regression_models(feature_info, len(df))

    elif task_type == "clustering":
        feature_info = analyze_feature_types(df, None)
        models = recommend_clustering_models(feature_info, len(df))

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return {
        "task_type": task_type,
        "dataset_size": df.shape[0],
        "feature_info": feature_info,
        "recommended_models": models
    }
