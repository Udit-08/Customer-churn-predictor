"""Supervised, rule-based churn risk groups for outreach prioritization.

These are not unsupervised customer types: the decision tree is trained using
Churn Value. Reported churn rates are calculated on a held-out test set.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

SEED = 42


def _describe(encoded_name, threshold):
    """Translate one encoded tree split into a plain-language rule."""
    name = encoded_name.split("__", 1)[-1]
    if name in NUMERIC_FEATURES:
        if name == "Tenure Months":
            cut = int(np.floor(threshold))
            return f"tenure ≤ {cut} months", f"tenure > {cut} months"
        cut = "$" + f"{threshold:,.0f}"
        return f"{name.lower()} ≤ {cut}", f"{name.lower()} > {cut}"

    for field in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        prefix = field + "_"
        if name.startswith(prefix):
            value = name[len(prefix):]
            return f"{field} is not '{value}'", f"{field} is '{value}'"
    return f"{name} is not the selected value", f"{name} is the selected value"


def _leaf_rules(tree, feature_names):
    nodes = tree.tree_
    rules = {}

    def walk(node, conditions):
        if nodes.children_left[node] == -1:
            rules[node] = " and ".join(conditions) if conditions else "All customers"
            return
        left, right = _describe(
            feature_names[nodes.feature[node]], nodes.threshold[node]
        )
        walk(nodes.children_left[node], conditions + [left])
        walk(nodes.children_right[node], conditions + [right])

    walk(0, [])
    return rules


def fit_risk_tree(data, depth=3, min_leaf=200):
    """Fit a shallow decision tree; all displayed test rates are held out.

    The encoder and imputer are fitted only on training rows. Churn Reason,
    Churn Label, Churn Score, CLTV and identifiers are not model inputs.
    """
    X = data[FEATURES].copy()
    X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")
    y = data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    numeric = SimpleImputer(strategy="median")
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)
    tree = DecisionTreeClassifier(
        max_depth=depth, min_samples_leaf=min_leaf, random_state=SEED
    )
    tree.fit(X_train_encoded, y_train)
    feature_names = preprocessor.get_feature_names_out()
    rules = _leaf_rules(tree, feature_names)

    train = (
        pd.DataFrame({"leaf": tree.apply(X_train_encoded), "y": y_train.to_numpy()})
        .groupby("leaf")["y"].agg(["size", "mean"])
    )
    test = (
        pd.DataFrame({"leaf": tree.apply(X_test_encoded), "y": y_test.to_numpy()})
        .groupby("leaf")["y"].agg(["size", "sum", "mean"])
    )
    result = pd.DataFrame({
        "Segment rule": pd.Series(rules),
        "Customers (train)": train["size"],
        "Churn rate (train)": train["mean"] * 100,
        "Customers (test)": test["size"],
        "Churners (test)": test["sum"],
        "Churn rate (test)": test["mean"] * 100,
    }).fillna(0)
    result["Share of test churners"] = result["Churners (test)"] / y_test.sum() * 100
    result = result.sort_values("Churn rate (train)", ascending=False).reset_index(drop=True)
    result.index = [f"R{index + 1}" for index in range(len(result))]
    return {
        "table": result,
        "test_rows": len(y_test),
        "test_churners": int(y_test.sum()),
        "base_rate": float(y_test.mean() * 100),
    }
