"""Time-to-churn analysis with censoring for currently active customers.

Tenure is treated as observed duration and Churn Value as the event indicator.
The Cox model excludes tenure and total charges from its predictors to avoid
using the observed duration (or a close proxy for it) to predict that duration.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import CATEGORICAL_FEATURES, TARGET

SEED = 42
PENALIZER = 0.05
HAZARD_BANDS = [0, 3, 6, 12, 24, 48, 73]


def _survival_classes():
    """Import lifelines only when the survival page is used."""
    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter
        from lifelines.utils import concordance_index
    except ImportError as exc:
        raise ImportError(
            "The survival page requires lifelines. Install the app requirements "
            "in the same Python environment used to run Streamlit."
        ) from exc
    return CoxPHFitter, KaplanMeierFitter, concordance_index


def _durations(data):
    return pd.to_numeric(data["Tenure Months"], errors="coerce").clip(lower=0.5)


def _fit_design(data):
    """Fit categorical vocabulary and monthly-charge scaling on training data."""
    categories = data[CATEGORICAL_FEATURES].copy()
    modes = {}
    for column in CATEGORICAL_FEATURES:
        mode = categories[column].dropna().astype(str).mode()
        modes[column] = mode.iloc[0] if len(mode) else "Not recorded"
        categories[column] = categories[column].fillna(modes[column]).astype(str)
    encoded = pd.get_dummies(categories, drop_first=True, dtype=float)
    charge = pd.to_numeric(data["Monthly Charges"], errors="coerce")
    charge_median = float(charge.median())
    charge = charge.fillna(charge_median)
    charge_mean = float(charge.mean())
    charge_std = float(charge.std())
    if not np.isfinite(charge_std) or charge_std == 0:
        charge_std = 1.0
    encoded["Monthly Charges"] = (charge - charge_mean) / charge_std
    params = {
        "columns": encoded.columns.tolist(),
        "modes": modes,
        "charge_median": charge_median,
        "charge_mean": charge_mean,
        "charge_std": charge_std,
    }
    return encoded, params


def _transform_design(data, params):
    categories = data[CATEGORICAL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        categories[column] = categories[column].fillna(params["modes"][column]).astype(str)
    encoded = pd.DataFrame(
        0.0, index=data.index, columns=params["columns"][:-1]
    )
    for field in CATEGORICAL_FEATURES:
        for index, value in categories[field].items():
            feature = f"{field}_{value}"
            if feature in encoded.columns:
                encoded.at[index, feature] = 1.0
    charge = pd.to_numeric(data["Monthly Charges"], errors="coerce")
    charge = charge.fillna(params["charge_median"])
    encoded["Monthly Charges"] = (
        charge - params["charge_mean"]
    ) / params["charge_std"]
    return encoded


def kaplan_meier(data, group_column=None, max_months=72):
    """Return retention estimates by month, optionally split by a field."""
    _, KaplanMeierFitter, _ = _survival_classes()
    times = np.arange(0, max_months + 1)
    groups = [("All customers", data)] if group_column is None else [
        (str(value), part) for value, part in data.groupby(group_column, dropna=False)
    ]
    rows = []
    for name, part in groups:
        if len(part) < 30:
            continue
        km = KaplanMeierFitter().fit(_durations(part), part[TARGET].astype(int))
        for month, value in zip(times, km.predict(times).values):
            rows.append({
                "Group": name, "Month": int(month), "Retained": float(value),
                "Customers": len(part),
            })
    return pd.DataFrame(rows)


def hazard_by_tenure(data):
    """Observed events divided by customer-month exposure in tenure bands."""
    duration = _durations(data)
    rows = []
    for start, end in zip(HAZARD_BANDS[:-1], HAZARD_BANDS[1:]):
        exposure = (np.minimum(duration, end) - start).clip(lower=0).sum()
        events = int((
            data[TARGET].eq(1) & duration.gt(start) & duration.le(end)
        ).sum())
        rows.append({
            "Months": f"{start}–{end if end < 73 else '72+'}",
            "Churn events": events,
            "Monthly churn rate": events / exposure if exposure else np.nan,
        })
    return pd.DataFrame(rows)


def _fit_cox(frame):
    CoxPHFitter, _, _ = _survival_classes()
    return CoxPHFitter(penalizer=PENALIZER).fit(frame, duration_col="T", event_col="E")


def fit_survival(data):
    """Evaluate a Cox model on holdout, then fit a full-data scoring model.

    Encoders, missing-value replacements, and numeric scaling for evaluation
    are fitted on the training split only. Active customers are right-censored.
    """
    _, KaplanMeierFitter, concordance_index = _survival_classes()
    train_idx, test_idx = train_test_split(
        np.arange(len(data)), test_size=0.20, random_state=SEED,
        stratify=data[TARGET].astype(int),
    )
    train_data, test_data = data.iloc[train_idx], data.iloc[test_idx]
    X_train, train_params = _fit_design(train_data)
    X_test = _transform_design(test_data, train_params)
    train = X_train.assign(
        T=_durations(train_data).to_numpy(), E=train_data[TARGET].astype(int).to_numpy()
    )
    test = X_test.assign(
        T=_durations(test_data).to_numpy(), E=test_data[TARGET].astype(int).to_numpy()
    )

    held_out = _fit_cox(train)
    risk = held_out.predict_partial_hazard(test[train_params["columns"]])
    c_index = float(concordance_index(test["T"], -risk, test["E"]))

    contract_columns = [c for c in train_params["columns"] if c.startswith("Contract_")]
    baseline = _fit_cox(train[contract_columns + ["T", "E"]])
    baseline_risk = baseline.predict_partial_hazard(test[contract_columns])
    c_contract = float(concordance_index(test["T"], -baseline_risk, test["E"]))

    predicted_12 = 1 - held_out.predict_survival_function(
        test[train_params["columns"]], times=[12]
    ).T.iloc[:, 0].to_numpy()
    scored = test.assign(predicted=predicted_12)
    labels = ["Group 1 (lowest)", "Group 2", "Group 3", "Group 4", "Group 5 (highest)"]
    scored["Risk group"] = pd.qcut(
        scored["predicted"].rank(method="first"), 5, labels=labels
    )
    calibration = []
    for group, part in scored.groupby("Risk group", observed=True):
        km = KaplanMeierFitter().fit(part["T"], part["E"])
        calibration.append({
            "Risk group": str(group), "Customers": len(part),
            "Predicted churn by month 12": float(part["predicted"].mean()),
            "Observed churn by month 12": float(1 - km.predict(12)),
        })

    # A 12-month AUC is evaluated only for customers whose 12-month status is known.
    known = scored[(scored["T"] >= 12) | ((scored["E"] == 1) & (scored["T"] < 12))]
    outcome = ((known["E"] == 1) & (known["T"] <= 12)).astype(int)
    from sklearn.metrics import roc_auc_score
    auc_12 = float(roc_auc_score(outcome, known["predicted"]))

    # Refit preprocessing and Cox model on all records for interactive curves.
    X_full, full_params = _fit_design(data)
    full_frame = X_full.assign(
        T=_durations(data).to_numpy(), E=data[TARGET].astype(int).to_numpy()
    )
    final = _fit_cox(full_frame)
    effects = (
        final.summary[["exp(coef)", "p"]]
        .rename(columns={"exp(coef)": "Hazard ratio", "p": "p-value"})
        .reset_index().rename(columns={"covariate": "Factor"})
    )
    return {
        "model": final,
        "columns": full_params["columns"],
        "design_params": full_params,
        "c_index": c_index,
        "c_index_contract_only": c_contract,
        "auc_12": auc_12,
        "calibration": pd.DataFrame(calibration),
        "effects": effects,
        "test_rows": len(test_data),
    }


def customer_curve(result, sample, horizon=72):
    """Return a customer's survival curve from the fitted full-data Cox model."""
    _, _, _ = _survival_classes()
    X = _transform_design(sample, result["design_params"])
    times = np.arange(0, horizon + 1)
    curve = result["model"].predict_survival_function(X, times=times).iloc[:, 0]
    return pd.DataFrame({"Month": times, "Retained": curve.to_numpy()})
