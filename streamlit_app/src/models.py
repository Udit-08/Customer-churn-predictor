import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import DMatrix, XGBClassifier

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET

SEED = 42


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def _make_model(pos_weight):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            min_child_weight=2, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight, eval_metric="auc",
            random_state=SEED, n_jobs=-1,
        )),
    ])


def metrics_at_threshold(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Churn precision": precision_score(y_true, predictions, zero_division=0),
        "Churn recall": recall_score(y_true, predictions, zero_division=0),
        "Churn F1": f1_score(y_true, predictions, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, probabilities),
        "Brier score": brier_score_loss(y_true, probabilities),
        "Confusion matrix": confusion_matrix(y_true, predictions, labels=[0, 1]),
        "Predictions": predictions,
    }


def fit_xgboost(data):
    X = data[FEATURES].copy()
    y = data[TARGET].astype(int)
    # data.clean_data already sets blank Total Charges to 0 for zero-tenure customers.
    # Any other blank is left as NaN so the pipeline's median imputer handles it
    # (fit on training rows only), instead of silently becoming 0 here.
    X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )

    # Compute the positive-class weight from each fold's training labels only.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_rows = []
    oof_train = np.zeros(len(X_train))  # out-of-fold scores, used for thresholds + calibration
    for fold, (fit_idx, val_idx) in enumerate(cv.split(X_train, y_train), start=1):
        y_fit = y_train.iloc[fit_idx]
        weight = (y_fit == 0).sum() / (y_fit == 1).sum()
        fold_model = _make_model(weight)
        fold_model.fit(X_train.iloc[fit_idx], y_fit)
        fold_probability = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]
        oof_train[val_idx] = fold_probability
        fold_metrics = metrics_at_threshold(y_train.iloc[val_idx], fold_probability)
        cv_rows.append({
            "Fold": fold,
            **{key: fold_metrics[key] for key in [
                "Accuracy", "Churn precision", "Churn recall", "Churn F1", "ROC AUC"
            ]},
        })
    cv_folds = pd.DataFrame(cv_rows).set_index("Fold")
    cv_summary = pd.DataFrame({"Mean": cv_folds.mean(), "Std": cv_folds.std(ddof=1)})

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = _make_model(pos_weight)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = metrics_at_threshold(y_test, probabilities, 0.5)
    baseline_brier = brier_score_loss(
        y_test, np.full(len(y_test), y_train.mean())
    )

    # Permute original input columns on test data to report customer-level features.
    importance = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=10,
        random_state=SEED, n_jobs=-1,
    )
    permutation = (
        pd.DataFrame({
            "Feature": FEATURES,
            "Mean ROC AUC drop": importance.importances_mean,
            "Std": importance.importances_std,
        })
        .sort_values("Mean ROC AUC drop", ascending=False)
        .reset_index(drop=True)
    )
    # Calibration: the class-weighted XGBoost score ranks customers well but its
    # numbers run high. An isotonic map fitted on TRAINING out-of-fold scores turns
    # the raw score into a realistic churn probability (the test set is not used).
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(oof_train, y_train)
    calibrated = calibrator.predict(probabilities)
    return {
        "model": model,
        "calibrator": calibrator,
        "oof_train_scores": oof_train,
        "y_train": y_train,
        "test_calibrated": calibrated,
        "calibration": {
            "Brier raw": brier_score_loss(y_test, probabilities),
            "Brier calibrated": brier_score_loss(y_test, calibrated),
            "ECE raw": expected_calibration_error(y_test, probabilities),
            "ECE calibrated": expected_calibration_error(y_test, calibrated),
            "curve_raw": calibration_points(y_test, probabilities),
            "curve_calibrated": calibration_points(y_test, calibrated),
        },
        "test_metrics": test_metrics,
        "test_probabilities": probabilities,
        "y_test": y_test,
        "cv_folds": cv_folds,
        "cv_summary": cv_summary,
        "permutation": permutation,
        "baseline_brier": baseline_brier,
        "positive_weight": pos_weight,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


# --------------------------------------------------------------------------- #
# Calibration, thresholds, gain, scoring and explanations
# --------------------------------------------------------------------------- #
def calibration_points(y_true, scores, bins=10):
    """Mean predicted vs observed churn rate in equal-sized score bins."""
    frame = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(scores)})
    frame["bin"] = pd.qcut(frame["p"].rank(method="first"), bins, labels=False)
    grouped = frame.groupby("bin").agg(predicted=("p", "mean"), observed=("y", "mean"),
                                       customers=("y", "size"))
    return grouped.reset_index(drop=True)


def expected_calibration_error(y_true, scores, bins=10):
    points = calibration_points(y_true, scores, bins)
    weights = points["customers"] / points["customers"].sum()
    return float((weights * (points["predicted"] - points["observed"]).abs()).sum())


def threshold_table(y_true, scores, thresholds=None, retention_value=0.0,
                    contact_cost=0.0, success_rate=1.0):
    """Precision/recall at many thresholds plus a simple net-benefit estimate.

    Net benefit = churners reached x success rate x value of keeping one customer
                  - customers contacted x cost of one contact.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    thresholds = np.arange(0.05, 0.951, 0.01) if thresholds is None else thresholds
    rows = []
    for threshold in thresholds:
        flagged = scores >= threshold
        tp = int((flagged & (y_true == 1)).sum())
        fp = int((flagged & (y_true == 0)).sum())
        fn = int((~flagged & (y_true == 1)).sum())
        rows.append({
            "Threshold": round(float(threshold), 2),
            "Flagged": tp + fp,
            "Precision": tp / (tp + fp) if tp + fp else np.nan,
            "Recall": tp / (tp + fn) if tp + fn else np.nan,
            "Missed churners": fn,
            "False alarms": fp,
            "Net benefit": tp * success_rate * retention_value - (tp + fp) * contact_cost,
        })
    return pd.DataFrame(rows)


def gain_curve(y_true, scores):
    """Share of all churners reached when contacting the top-scored customers first."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    sorted_y = np.asarray(y_true)[order]
    contacted = np.arange(1, len(sorted_y) + 1) / len(sorted_y)
    captured = np.cumsum(sorted_y) / sorted_y.sum()
    keep = np.unique(np.linspace(0, len(sorted_y) - 1, 200).astype(int))
    return pd.DataFrame({"Customers contacted": contacted[keep],
                         "Churners reached": captured[keep]})


def score_all_customers(data, calibrator):
    """Out-of-fold churn scores for EVERY customer.

    Each customer is scored by a model that never saw that customer, so the
    numbers are honest. Returns (raw scores, calibrated probabilities).
    """
    X = data[FEATURES].copy()
    X["Total Charges"] = pd.to_numeric(X["Total Charges"], errors="coerce")
    y = data[TARGET].astype(int)
    raw = np.zeros(len(X))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    for fit_idx, val_idx in cv.split(X, y):
        y_fit = y.iloc[fit_idx]
        weight = (y_fit == 0).sum() / (y_fit == 1).sum()
        model = _make_model(weight).fit(X.iloc[fit_idx], y_fit)
        raw[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    return raw, calibrator.predict(raw)


def explain_customer(model, sample, top_n=8):
    """Per-customer feature contributions (XGBoost's built-in TreeSHAP).

    Positive numbers push the churn score up, negative numbers push it down.
    One-hot columns are folded back into the original customer field.
    """
    prep, booster = model.named_steps["prep"], model.named_steps["model"].get_booster()
    encoded = prep.transform(sample[FEATURES])
    contributions = booster.predict(DMatrix(encoded), pred_contribs=True)[0][:-1]
    owners = []
    for name in prep.get_feature_names_out():
        field = name.split("__", 1)[1]
        owner = next((c for c in sorted(FEATURES, key=len, reverse=True)
                      if field == c or field.startswith(c + "_")), field)
        owners.append(owner)
    table = (pd.DataFrame({"Field": owners, "Effect": contributions})
             .groupby("Field", as_index=False)["Effect"].sum())
    table["Customer value"] = table["Field"].map(
        lambda f: str(sample.iloc[0][f]) if f in sample.columns else "")
    table["abs"] = table["Effect"].abs()
    return table.sort_values("abs", ascending=False).head(top_n).drop(columns="abs")
