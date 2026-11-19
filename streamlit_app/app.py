import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans

from src.clustering import (
    compare_segmentations, dendrogram_data, fit_clustering, fit_two_stage,
    natural_structure, pca_map, prepare_matrix, rule_labels, rule_table,
    silhouette_by_label,
)
from src.data import (
    CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, TARGET,
    DatasetError, default_workbook_path, load_workbook,
)
from src.models import (
    explain_customer, fit_xgboost, gain_curve, metrics_at_threshold,
    score_all_customers, threshold_table,
)
from src.risk_segments import fit_risk_tree
from src.ui import (
    AMBER, BLUE, CORAL, GREY, NAVY, PALETTE, TEAL, card, insight, kpi, money, note,
    page_header, setup_page, show, table,
)

setup_page()

FLAG = 0.50  # reference decision threshold on the raw XGBoost score
TENURE_LABELS = ["0–12", "13–24", "25–36", "37–48", "49–60", "61+"]
CHARGE_LABELS = ["Under $35", "$35–55", "$55–75", "$75–95", "$95+"]
PAGES = ["Overview", "Explore", "Prediction", "Segments", "Action center", "Conclusions"]


# --------------------------------------------------------------------------- #
# Cached computation
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data(uploaded_bytes, file_name):
    source = uploaded_bytes if uploaded_bytes is not None else default_workbook_path()
    frame = load_workbook(source)
    frame["Tenure group"] = pd.cut(frame["Tenure Months"], [-1, 12, 24, 36, 48, 60, 100],
                                   labels=TENURE_LABELS)
    frame["Monthly charge band"] = pd.cut(frame["Monthly Charges"], [0, 35, 55, 75, 95, 200],
                                          labels=CHARGE_LABELS)
    return frame


@st.cache_resource(show_spinner="Training XGBoost with five-fold validation…")
def get_models(data):
    return fit_xgboost(data)


@st.cache_resource(show_spinner="Clustering all customers…")
def get_clusters(data, k, eps, min_samples, collapse_redundant, extended_checks):
    return fit_clustering(
        data, k, eps, min_samples, collapse_redundant=collapse_redundant,
        include_k_sweep=extended_checks, include_stability=extended_checks,
    )


@st.cache_resource(show_spinner="Segmenting internet customers…")
def get_two_stage(data, k, collapse_redundant, extended_checks):
    return fit_two_stage(
        data, k, collapse_redundant,
        include_k_sweep=extended_checks,
        include_stability=extended_checks,
    )


@st.cache_data(show_spinner="Checking for natural groups…")
def get_structure(data):
    return natural_structure(data)


@st.cache_data(show_spinner=False)
def get_label_quality(data, labels):
    return silhouette_by_label(data, labels)


@st.cache_data(show_spinner=False)
def get_map(data, collapse_redundant):
    return pca_map(data, collapse_redundant)


@st.cache_data(show_spinner=False)
def get_kmeans_labels(data, k=3):
    matrix = prepare_matrix(data)
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(matrix)


@st.cache_data(show_spinner="Fitting the supervised risk tree…")
def get_risk_tree(data, depth, min_leaf):
    return fit_risk_tree(data, depth, min_leaf)


@st.cache_resource(show_spinner="Scoring every customer…")
def get_scored(data):
    """Every customer with an out-of-fold churn probability and a default segment."""
    result = get_models(data)
    raw, calibrated = score_all_customers(data, result["calibrator"])
    segmentation = get_two_stage(data, 3, False, False)
    profile = segmentation["profile"][0]
    scored = data.copy()
    scored["Segment"] = pd.Series(segmentation["labels"]).map(profile["Segment"]).values
    scored["Raw score"] = raw
    scored["Churn probability"] = calibrated
    scored["Flagged"] = raw >= FLAG
    scored["Active"] = scored[TARGET].eq(0)
    scored["Revenue at risk"] = np.where(
        scored["Active"], calibrated * scored["Monthly Charges"], 0.0)
    scored["Risk"] = np.where(~scored["Flagged"], "Low",
                              np.where(calibrated >= 0.5, "High", "Medium"))
    return scored


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def churn_bar(frame, column, title, overall=None):
    summary = (frame.groupby(column, dropna=False, observed=True)[TARGET]
               .agg(rate="mean", customers="count").reset_index())
    summary["rate"] *= 100
    if not isinstance(frame[column].dtype, pd.CategoricalDtype):
        summary = summary.sort_values("rate", ascending=False)
    summary[column] = summary[column].astype(str)
    fig = px.bar(summary, x=column, y="rate", text="rate", color="rate",
                 color_continuous_scale=["#8ED1C6", "#F0C36A", CORAL], hover_data={"customers": True},
                 title=title)
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="Churn rate (%)", xaxis_title="")
    fig.update_yaxes(range=[0, max(summary["rate"].max() * 1.25, 10)])
    if overall is not None:
        fig.add_hline(y=overall * 100, line_dash="dot", line_color=GREY,
                      annotation_text=f"All customers {overall:.1%}",
                      annotation_position="top right", annotation_font_color=GREY)
    return fig


def risk_pill_class(risk):
    return {"High": "pill-high", "Medium": "pill-mid"}.get(risk, "pill-low")


def format_profile(profile):
    shown = profile.rename(columns={
        "Churn_Rate": "Churn rate", "Median_Tenure": "Median tenure (mo)",
        "Avg_Monthly_Charges": "Avg monthly", "Avg_Total_Charges": "Avg total",
        "Share_%": "Share of customers", "Churn_Lift": "Churn lift",
        "Churners_Captured_%": "Share of churners", "Dependents_Yes_%": "Dependents",
        "Senior_Yes_%": "Seniors", "Electronic_Check_%": "Electronic check",
        "Avg_Addon_Services": "Avg add-ons",
    }).copy()
    for column in ["Churn rate", "Share of customers", "Share of churners",
                   "Dependents", "Seniors", "Electronic check"]:
        shown[column] = shown[column].map(lambda v: f"{v:.1f}%")
    shown["Churn lift"] = shown["Churn lift"].map(lambda v: f"{v:.2f}×")
    for column in ["Avg monthly", "Avg total"]:
        shown[column] = shown[column].map(money)
    shown["Avg add-ons"] = shown["Avg add-ons"].map(lambda v: f"{v:.1f}")
    shown["Median tenure (mo)"] = shown["Median tenure (mo)"].map(lambda v: f"{v:.0f}")
    shown.index = [("Noise" if i == -1 else f"#{i}") for i in shown.index]
    first = ["Segment", "Customers", "Share of customers", "Churn rate", "Churn lift",
             "Share of churners"]
    return shown[first + [c for c in shown.columns if c not in first]]


def cluster_feature_mix(data, labels, cluster, feature):
    members = data.iloc[np.flatnonzero(np.asarray(labels) == cluster)]
    values = members[feature].fillna("Not recorded").astype(str)
    counts = values.value_counts(dropna=False)
    return pd.DataFrame({
        "Value": counts.index,
        "Customers": counts.values,
        "Share within cluster (%)": counts.values / max(len(members), 1) * 100,
    })


def cluster_reason_mix(data, labels, cluster):
    members = data.iloc[
        np.flatnonzero((np.asarray(labels) == cluster) & data[TARGET].eq(1).to_numpy())
    ]
    if "Churn Reason" not in members.columns:
        return pd.DataFrame(columns=["Reason", "Churned customers", "Share of cluster churners (%)"])
    reasons = members["Churn Reason"].fillna("Not recorded").astype(str).str.strip()
    reasons = reasons.replace("", "Not recorded")
    counts = reasons.value_counts(dropna=False)
    return pd.DataFrame({
        "Reason": counts.index,
        "Churned customers": counts.values,
        "Share of cluster churners (%)": counts.values / max(len(members), 1) * 100,
    })


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def render_overview(data, scored):
    page_header("Customer retention analytics", "Churn at a glance",
                "Who leaves, when, and which customers to contact first. All numbers are "
                "recomputed from the workbook.")
    churned = int(data[TARGET].sum())
    total = len(data)
    rate = churned / total
    active = scored[scored["Active"]]
    flagged = active[active["Flagged"]]
    cols = st.columns(5)
    kpi(cols[0], "Customers", f"{total:,}", f"{len(active):,} still active")
    kpi(cols[1], "Churn rate", f"{rate:.1%}", f"{churned:,} customers left")
    kpi(cols[2], "Revenue lost / month", money(data.loc[data[TARGET].eq(1), "Monthly Charges"].sum()),
        "from customers who left")
    kpi(cols[3], "Active, flagged", f"{len(flagged):,}", f"{len(flagged) / max(len(active), 1):.0%} of active customers")
    kpi(cols[4], "Revenue at risk", money(active["Revenue at risk"].sum()), "per month, active customers")
    st.write("")

    contract = data.groupby("Contract")[TARGET].mean()
    first_year = data.loc[data["Tenure Months"] <= 12, TARGET].mean()
    later = data.loc[data["Tenure Months"] > 12, TARGET].mean()
    internet = data.groupby("Internet Service")[TARGET].mean()
    left, right = st.columns([1, 1.25])
    with left:
        st.subheader("What stands out")
        insight(f"<b>Contract:</b> month-to-month customers churn at "
                f"<b>{contract['Month-to-month']:.0%}</b>, against "
                f"<b>{contract['Two year']:.0%}</b> on two-year contracts.")
        insight(f"<b>Tenure:</b> customers in their first year churn at "
                f"<b>{first_year:.0%}</b>, versus <b>{later:.0%}</b> after that.")
        insight(f"<b>Internet:</b> fiber optic customers churn at "
                f"<b>{internet['Fiber optic']:.0%}</b>, DSL at <b>{internet['DSL']:.0%}</b>, "
                f"and customers without internet at <b>{internet['No']:.0%}</b>.")
        note("These are associations in one dataset, not proof that a feature causes churn.")
    with right:
        show(churn_bar(data, "Contract", "Churn rate by contract", rate), key="ov_contract")

    left, right = st.columns(2)
    with left:
        show(churn_bar(data, "Tenure group", "Churn rate by tenure (months)", rate), key="ov_tenure")
    with right:
        show(churn_bar(data, "Monthly charge band", "Churn rate by monthly bill", rate), key="ov_charge")

    st.subheader("Where the retention money is")
    seg = (active.groupby("Segment")
           .agg(customers=("Segment", "size"), risk=("Revenue at risk", "sum"),
                prob=("Churn probability", "mean"))
           .sort_values("risk", ascending=False).reset_index())
    cols = st.columns(max(len(seg), 1))
    for col, (_, row) in zip(cols, seg.iterrows()):
        with col:
            card(row["Segment"], money(row["risk"]),
                 f"monthly revenue at risk · {row['customers']:,} active customers",
                 f"avg churn probability {row['prob']:.0%}",
                 "pill-high" if row["prob"] >= 0.3 else "pill-low")
    st.caption("Segments use two stages: customers without internet are set aside, then "
               "K-Means (k=3) groups customers with internet. Revenue at risk = estimated churn "
               "probability × monthly charge, summed over customers who have not churned.")


def render_explore(data):
    page_header("Explore", "Where does churn vary?",
                "Filter the customer base and watch every number and chart update.")
    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.1, 1.3])
    contract = f1.multiselect("Contract", sorted(data["Contract"].unique()), placeholder="All")
    internet = f2.multiselect("Internet service", sorted(data["Internet Service"].unique()),
                              placeholder="All")
    payment = f3.multiselect("Payment method", sorted(data["Payment Method"].unique()),
                             placeholder="All")
    tenure = f4.slider("Tenure (months)", 0, int(data["Tenure Months"].max()),
                       (0, int(data["Tenure Months"].max())))
    view = data[data["Tenure Months"].between(*tenure)]
    for column, chosen in [("Contract", contract), ("Internet Service", internet),
                           ("Payment Method", payment)]:
        if chosen:
            view = view[view[column].isin(chosen)]
    if len(view) < 30:
        st.warning("Fewer than 30 customers match these filters. Loosen them to see reliable rates.")
        return
    overall = data[TARGET].mean()
    rate = view[TARGET].mean()
    cols = st.columns(5)
    diff = (rate - overall) * 100
    kpi(cols[0], "Customers", f"{len(view):,}", f"{len(view) / len(data):.0%} of all customers")
    kpi(cols[1], "Churn rate", f"{rate:.1%}", f"{diff:+.1f} pts vs all customers",
        "bad" if diff > 0.5 else "good" if diff < -0.5 else "")
    kpi(cols[2], "Avg monthly charge", money(view["Monthly Charges"].mean()))
    kpi(cols[3], "Avg tenure", f"{view['Tenure Months'].mean():.0f} mo")
    kpi(cols[4], "Revenue lost / month", money(view.loc[view[TARGET].eq(1), "Monthly Charges"].sum()))
    st.write("")

    left, right = st.columns(2)
    with left:
        show(churn_bar(view, "Contract", "Churn by contract", overall), key="ex_contract")
    with right:
        show(churn_bar(view, "Tenure group", "Churn by tenure (months)", overall), key="ex_tenure")
    left, right = st.columns(2)
    with left:
        show(churn_bar(view, "Monthly charge band", "Churn by monthly bill", overall), key="ex_charge")
    with right:
        service = st.selectbox("Service", ["Internet Service", "Online Security", "Tech Support",
                                           "Online Backup", "Device Protection", "Streaming TV",
                                           "Streaming Movies", "Phone Service", "Multiple Lines",
                                           "Payment Method", "Paperless Billing"])
        show(churn_bar(view, service, f"Churn by {service.lower()}", overall), key="ex_service")

    st.subheader("Two-way view")
    fields = ["Contract", "Tenure group", "Internet Service", "Payment Method",
              "Monthly charge band", "Senior Citizen", "Dependents", "Partner",
              "Tech Support", "Online Security"]
    a, b = st.columns(2)
    row_field = a.selectbox("Rows", fields, index=0)
    col_field = b.selectbox("Columns", fields, index=1)
    if row_field == col_field:
        st.info("Pick two different fields.")
    else:
        pivot = view.pivot_table(index=row_field, columns=col_field, values=TARGET,
                                 aggfunc=["mean", "count"], observed=True)
        rates = (pivot["mean"] * 100).where(pivot["count"] >= 30)
        fig = px.imshow(rates, text_auto=".0f", aspect="auto", color_continuous_scale=["#E6F6F3", CORAL],
                        labels=dict(color="Churn %"), title="Churn rate (%) — cells under 30 customers hidden")
        fig.update_xaxes(side="bottom", title="")
        fig.update_yaxes(title="")
        show(fig, key="ex_heat")

    st.subheader("Why customers say they left")
    reasons = (view.loc[view[TARGET].eq(1), "Churn Reason"].dropna().value_counts()
               .rename_axis("Reason").reset_index(name="Customers").head(12)
               if "Churn Reason" in view.columns else pd.DataFrame())
    if len(reasons):
        fig = px.bar(reasons.sort_values("Customers"), x="Customers", y="Reason", orientation="h")
        fig.update_traces(marker_color=NAVY)
        fig.update_layout(yaxis_title="", xaxis_title="Churned customers", height=420)
        show(fig, key="ex_reasons")
        st.caption("Stated reasons are collected after a customer leaves, so they describe churn "
                   "but are never used to predict it.")


def render_prediction(data, scored):
    page_header("Supervised learning", "Churn prediction with XGBoost",
                "One model scores every customer. Test results come from a held-out 20% of "
                "customers; thresholds and calibration are learned from training data only.")
    result = get_models(data)
    tabs = st.tabs(["Performance", "Threshold and cost", "Probabilities", "What drives it",
                    "Score a customer"])

    # ---- Performance ----
    with tabs[0]:
        threshold = st.slider("Decision threshold on the raw model score", 0.10, 0.90, FLAG, 0.05,
                              help="Customers scoring at or above this value are flagged. "
                                   "The reference choice is 0.50.")
        st.caption(
            "These test-set scores are for exploration. Do not choose a threshold from "
            "the test set; use the training-only out-of-fold analysis in the next tab."
        )
        m = metrics_at_threshold(result["y_test"], result["test_probabilities"], threshold)
        cols = st.columns(5)
        for col, label in zip(cols, ["Churn recall", "Churn precision", "Churn F1", "ROC AUC", "Accuracy"]):
            col.metric(label, f"{m[label]:.3f}")
        tn, fp, fn, tp = m["Confusion matrix"].ravel()
        insight(f"At threshold <b>{threshold:.2f}</b> the model finds <b>{tp}</b> of "
                f"<b>{tp + fn}</b> churners in the test set (<b>{fn}</b> missed) and flags "
                f"<b>{fp}</b> customers who stayed.")
        left, right = st.columns(2)
        with left:
            fig = px.imshow(m["Confusion matrix"], text_auto=True, aspect="auto",
                            color_continuous_scale=["#EAF6F4", TEAL],
                            x=["Predicted: stayed", "Predicted: churned"],
                            y=["Actual: stayed", "Actual: churned"], title="Confusion matrix (test set)")
            fig.update_layout(coloraxis_showscale=False)
            show(fig, key="pf_cm")
        with right:
            gain = gain_curve(result["y_test"], result["test_probabilities"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gain["Customers contacted"], y=gain["Churners reached"],
                                     mode="lines", name="XGBoost ranking", line=dict(color=TEAL, width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random order",
                                     line=dict(color=GREY, dash="dot")))
            fig.update_layout(title="Gain curve (test set)", xaxis_tickformat=".0%",
                              yaxis_tickformat=".0%", xaxis_title="Share of customers contacted (highest score first)",
                              yaxis_title="Share of churners reached")
            show(fig, key="pf_gain")
        top20 = float(np.interp(0.2, gain["Customers contacted"], gain["Churners reached"]))
        st.caption(f"Contacting the top 20% of customers by score reaches about {top20:.0%} of churners.")
        st.subheader("Five-fold cross-validation (training data only)")
        cv = result["cv_summary"].copy()
        cv["Mean ± std"] = cv.apply(lambda r: f"{r['Mean']:.3f} ± {r['Std']:.3f}", axis=1)
        table(cv[["Mean ± std"]])
        with st.expander("Model settings and safeguards"):
            st.markdown(
                "- **XGBoost:** 300 trees, learning rate 0.05, depth 3, min child weight 2, "
                "row/column sampling 0.8, random state 42, class weight from training labels.\n"
                "- **Split:** 80/20 stratified, random state 42. Preprocessing is inside the "
                "pipeline and fitted on training rows only; no early stopping, no test-set tuning.\n"
                "- **Inputs:** " + ", ".join(FEATURES) + ".\n"
                "- **Excluded (leakage):** Churn Value (target), Churn Label, Churn Reason, "
                "Churn Score, CLTV, CustomerID and location fields.")

    # ---- Threshold and cost ----
    with tabs[1]:
        st.write("Choose a threshold by what a missed churner and a wasted contact cost. "
                 "Curves use out-of-fold scores on the **training** data, so the test set stays untouched.")
        st.caption(
            "The starting values below are examples for exploration, not measured "
            "business costs. Replace them with estimates from your organization."
        )
        a, b, c = st.columns(3)
        value = a.number_input("Value of keeping one customer ($)", 0, 10000, 500, 50)
        cost = b.number_input("Cost of contacting one customer ($)", 0, 1000, 20, 5)
        success = c.slider("Chance a contacted churner is saved", 0.05, 1.0, 0.30, 0.05)
        tt = threshold_table(result["y_train"], result["oof_train_scores"],
                             retention_value=value, contact_cost=cost, success_rate=success)
        tt["Net benefit per 1,000"] = tt["Net benefit"] / len(result["y_train"]) * 1000
        best = tt.loc[tt["Net benefit per 1,000"].idxmax()]
        at_ref = tt.loc[(tt["Threshold"] - FLAG).abs().idxmin()]
        cols = st.columns(4)
        cols[0].metric("Best threshold", f"{best['Threshold']:.2f}")
        cols[1].metric("Net benefit per 1,000 (best)", money(best["Net benefit per 1,000"]))
        cols[2].metric("Net benefit per 1,000 (at 0.50)", money(at_ref["Net benefit per 1,000"]))
        cols[3].metric("Recall at best", f"{best['Recall']:.0%}")
        left, right = st.columns(2)
        with left:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tt["Threshold"], y=tt["Recall"], name="Recall",
                                     line=dict(color=TEAL, width=3)))
            fig.add_trace(go.Scatter(x=tt["Threshold"], y=tt["Precision"], name="Precision",
                                     line=dict(color=CORAL, width=3)))
            fig.add_vline(x=FLAG, line_dash="dot", line_color=GREY)
            fig.update_layout(title="Precision and recall by threshold", yaxis_tickformat=".0%",
                              xaxis_title="Threshold")
            show(fig, key="th_pr")
        with right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tt["Threshold"], y=tt["Net benefit per 1,000"],
                                     line=dict(color=NAVY, width=3), name="Net benefit"))
            fig.add_vline(x=best["Threshold"], line_color=TEAL, annotation_text="best")
            fig.add_vline(x=FLAG, line_dash="dot", line_color=GREY, annotation_text="0.50")
            fig.update_layout(title="Estimated net benefit per 1,000 customers", yaxis_title="$",
                              xaxis_title="Threshold", showlegend=False)
            show(fig, key="th_nb")
        test_best = metrics_at_threshold(result["y_test"], result["test_probabilities"], best["Threshold"])
        st.caption(f"With these costs the best threshold is {best['Threshold']:.2f}. For reference, "
                   f"on the test set that gives recall {test_best['Churn recall']:.2f} and precision "
                   f"{test_best['Churn precision']:.2f}; the project keeps 0.50 as its reference "
                   "because recall on churners is the priority.")

    # ---- Probabilities ----
    with tabs[2]:
        cal = result["calibration"]
        base = float(result["baseline_brier"])
        cols = st.columns(4)
        cols[0].metric("Brier score, raw", f"{cal['Brier raw']:.3f}")
        cols[1].metric("Brier score, calibrated", f"{cal['Brier calibrated']:.3f}")
        cols[2].metric("Calibration error, raw", f"{cal['ECE raw']:.3f}")
        cols[3].metric("Calibration error, calibrated", f"{cal['ECE calibrated']:.3f}")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect",
                                 line=dict(color=GREY, dash="dot")))
        for name, key, color in [("Raw XGBoost score", "curve_raw", CORAL),
                                 ("Calibrated probability", "curve_calibrated", TEAL)]:
            pts = cal[key]
            fig.add_trace(go.Scatter(x=pts["predicted"], y=pts["observed"], mode="lines+markers",
                                     name=name, line=dict(color=color, width=3)))
        fig.update_layout(title="Predicted vs observed churn rate (test set, 10 equal bins)",
                          xaxis_title="Predicted", yaxis_title="Observed churn rate",
                          xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=440)
        show(fig, key="pr_cal")
        ref_prob = float(result["calibrator"].predict([FLAG])[0])
        insight(f"The raw score is built for ranking and runs high because the model is class-weighted. "
                f"Calibration is learned from training out-of-fold scores; raw 0.50 maps to about "
                f"<b>{ref_prob:.0%}</b> calibrated probability. The test calibration curve shows "
                f"how observed rates compare in score bins. A constant training-rate prediction has "
                f"Brier <b>{base:.3f}</b>.")
        st.caption("The calibration map is learned from training out-of-fold scores, so the test set "
                   "is still an honest check.")

    # ---- Drivers ----
    with tabs[3]:
        imp = result["permutation"].head(12).sort_values("Mean ROC AUC drop")
        fig = px.bar(imp, x="Mean ROC AUC drop", y="Feature", orientation="h", error_x="Std",
                     title="Permutation importance (test set)")
        fig.update_traces(marker_color=TEAL)
        fig.update_layout(yaxis_title="", xaxis_title="Drop in ROC AUC when the field is shuffled", height=460)
        show(fig, key="dr_imp")
        top = result["permutation"].head(3)["Feature"].tolist()
        insight(f"The model leans most on <b>{top[0]}</b>, then <b>{top[1]}</b> and <b>{top[2]}</b>.")
        st.caption("These describe what the model uses, not what causes churn. Shuffling a field "
                   "breaks its link to the outcome; a large drop means the model relied on it.")

    # ---- Score a customer ----
    with tabs[4]:
        st.write("Describe a customer to get a calibrated churn probability and the reasons behind it.")
        values = {}
        cols = st.columns(4)
        for i, column in enumerate(CATEGORICAL_FEATURES):
            options = sorted(data[column].dropna().astype(str).unique().tolist())
            default = data[column].mode().iloc[0]
            values[column] = cols[i % 4].selectbox(column, options, index=options.index(str(default)),
                                                   key=f"sc_{column}")
        n1, n2, n3 = st.columns(3)
        values["Tenure Months"] = n1.slider("Tenure (months)", 0, 72, 12, key="sc_tenure")
        values["Monthly Charges"] = n2.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.5, key="sc_monthly")
        override = n3.checkbox("Enter total charges manually", key="sc_override")
        estimated = float(values["Tenure Months"] * values["Monthly Charges"])
        values["Total Charges"] = (n3.number_input("Total charges ($)", 0.0, 12000.0, estimated, 10.0,
                                                   key="sc_total") if override else estimated)
        if not override:
            n3.caption(f"Total charges estimated as tenure × monthly: {money(estimated)}")
        sample = pd.DataFrame([{c: values[c] for c in FEATURES}])
        raw = float(result["model"].predict_proba(sample)[0, 1])
        prob = float(result["calibrator"].predict([raw])[0])
        flagged = raw >= FLAG
        band = "Low" if not flagged else ("High" if prob >= 0.5 else "Medium")
        st.write("")
        left, right = st.columns([1, 1.5])
        with left:
            card("Estimated churn probability", f"{prob:.0%}",
                 f"raw model score {raw:.2f} · {'flagged' if flagged else 'not flagged'} at the 0.50 threshold",
                 f"{band} risk", risk_pill_class(band))
        with right:
            why = explain_customer(result["model"], sample)
            why["Direction"] = np.where(why["Effect"] > 0, "Raises churn risk", "Lowers churn risk")
            why["Label"] = why["Field"] + ": " + why["Customer value"]
            fig = px.bar(why.iloc[::-1], x="Effect", y="Label", orientation="h", color="Direction",
                         color_discrete_map={"Raises churn risk": CORAL, "Lowers churn risk": TEAL},
                         title="Top reasons for this score")
            fig.update_layout(yaxis_title="", xaxis_title="Effect on the score (log-odds)", height=380)
            show(fig, key="sc_why")
        st.caption("Reasons come from XGBoost's built-in tree explanations. They describe this model's "
                   "behaviour for this customer, not a guaranteed cause.")


def render_choosing_k(ks, stable, internet_only):
    if ks is None:
        st.info("Turn on “Run k-selection and stability checks” in the settings above.")
        return
    if internet_only:
        st.caption(
            "These scores are for customers with internet only; don’t compare them "
            "directly with scores calculated on all customers."
        )
    left, right = st.columns(2)
    silhouette_fig = px.line(
        ks.reset_index(), x="k", y="Silhouette", markers=True,
        title="Silhouette by k (higher is generally better)",
    )
    silhouette_fig.update_traces(line_color=TEAL)
    ch_fig = px.line(
        ks.reset_index(), x="k", y="Calinski-Harabasz", markers=True,
        title="Calinski–Harabasz by k (higher is generally better)",
    )
    ch_fig.update_traces(line_color=BLUE)
    with left:
        show(silhouette_fig, key=f"k_sil_{internet_only}")
    with right:
        show(ch_fig, key=f"k_ch_{internet_only}")
    table(ks.style.format("{:.3f}"))
    if stable is not None:
        cols = st.columns(4)
        cols[0].metric("Seed ARI (mean)", f"{stable['seed_ari_mean']:.3f}")
        cols[1].metric("Seed ARI (min)", f"{stable['seed_ari_min']:.3f}")
        cols[2].metric("Bootstrap ARI (mean)", f"{stable['bootstrap_ari_mean']:.3f}")
        cols[3].metric("Bootstrap ARI (min)", f"{stable['bootstrap_ari_min']:.3f}")
        st.caption(
            "These checks help assess sensitivity to k and resampling. They don’t "
            "prove that the groups are natural or useful for a particular action."
        )


def render_how_real(data, labels, names, current_name):
    st.markdown("**1. Does density-based clustering find natural groups?**")
    structure = get_structure(data)
    if structure is None:
        st.info("This check needs scikit-learn 1.3 or newer for HDBSCAN.")
    else:
        table(structure.style.format({"Noise (%)": "{:.1f}"}))
        if structure["Clusters found"].eq(2).all():
            st.caption(
                "HDBSCAN returned two groups at each tested minimum size. The table "
                "shows their sizes and noise; compare those sizes with the internet "
                "and no-internet counts."
            )
        else:
            st.caption(
                "HDBSCAN results vary with its minimum cluster size. Use the table to "
                "see how many groups and noise points it returns at each setting."
            )
        st.caption(
            "HDBSCAN runs on ten PCA components of the customer/service/contract/"
            "tenure/billing features. It is a structural check, not proof that the "
            "groups are universal customer types."
        )

    st.markdown("**2. How separated is each displayed segment?**")
    quality = get_label_quality(data, labels)
    quality_table = pd.DataFrame({
        "Segment": [names.get(int(cluster), f"#{cluster}") for cluster in quality.index],
        "Average silhouette": quality.values,
    })
    fig = px.bar(
        quality_table.sort_values("Average silhouette"),
        x="Average silhouette", y="Segment", orientation="h",
        text="Average silhouette", title="Silhouette by segment",
    )
    fig.update_traces(texttemplate="%{text:.2f}", marker_color=TEAL)
    fig.add_vline(x=0, line_dash="dot", line_color=GREY)
    fig.update_layout(yaxis_title="", xaxis_title="Average silhouette in transformed feature space")
    show(fig, key="real_silhouette")
    st.caption(
        "Scores nearer 1 indicate tighter, more separated groups; values near 0 "
        "indicate overlap. A silhouette score describes geometry, not business value."
    )

    st.markdown("**3. Compare segmentations after grouping**")
    other_name = (
        "Single-stage K-Means (3)"
        if current_name.startswith("Two-stage")
        else "Two-stage (4)"
    )
    other_labels = (
        get_kmeans_labels(data, 3)
        if current_name.startswith("Two-stage")
        else get_two_stage(data, 3, False, False)["labels"]
    )
    comparison = compare_segmentations(data, {
        current_name: labels,
        other_name: other_labels,
        "Simple rules (4)": rule_labels(data),
    })
    table(comparison.style.format({
        "Highest churn (%)": "{:.1f}",
        "Lowest churn (%)": "{:.1f}",
        "Customers in top segment": "{:,.0f}",
        "Churners in top segment (%)": "{:.1f}",
        "Link to churn (NMI)": "{:.3f}",
    }))
    st.caption(
        "Churn rates and NMI are post-hoc comparisons with the outcome; they do not "
        "determine the unsupervised groups or measure whether groups are naturally real."
    )
    st.markdown("**Simple rule groups**")
    rule_summary = rule_table(data)
    table(rule_summary, hide_index=True, column_config={
        "Customers": st.column_config.NumberColumn(format="%d"),
        "Churn rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Share of churners": st.column_config.NumberColumn(format="%.1f%%"),
    })


def render_risk_tree(data):
    note(
        "This is a <b>supervised risk grouping</b>, not unsupervised segmentation. "
        "The tree learns rules using Churn Value on training customers; every churn "
        "rate labeled test is measured on the held-out test set."
    )
    left, right = st.columns(2)
    depth = left.slider("Risk tree depth", 2, 4, 3, key="risk_tree_depth")
    min_leaf = right.slider(
        "Minimum training customers in a rule group", 100, 500, 200, 50,
        key="risk_tree_min_leaf",
    )
    result = get_risk_tree(data, depth, min_leaf)
    risk_table = result["table"]
    top = risk_table.sort_values("Churn rate (test)", ascending=False).iloc[0]
    cols = st.columns(4)
    kpi(cols[0], "Risk groups", len(risk_table))
    kpi(cols[1], "Top group test churn", f"{top['Churn rate (test)']:.1f}%",
        f"{int(top['Customers (test)']):,} test customers")
    kpi(cols[2], "Test churners in top group", f"{top['Share of test churners']:.1f}%")
    kpi(cols[3], "Overall test churn rate", f"{result['base_rate']:.1f}%")

    plot_data = risk_table.assign(Group=risk_table.index)
    fig = px.bar(
        plot_data, x="Group", y="Churn rate (test)", text="Churn rate (test)",
        hover_data={"Segment rule": True, "Customers (test)": True},
        title="Risk group churn rates on held-out customers",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", marker_color=CORAL, cliponaxis=False)
    fig.add_hline(
        y=result["base_rate"], line_dash="dot", line_color=GREY,
        annotation_text=f"Overall {result['base_rate']:.1f}%",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Churn rate (%)")
    show(fig, key="risk_tree_rates")
    table(risk_table[[
        "Segment rule", "Customers (train)", "Churn rate (train)",
        "Customers (test)", "Churners (test)", "Churn rate (test)",
        "Share of test churners",
    ]], column_config={
        "Churn rate (train)": st.column_config.NumberColumn(format="%.1f%%"),
        "Churn rate (test)": st.column_config.NumberColumn(format="%.1f%%"),
        "Share of test churners": st.column_config.NumberColumn(format="%.1f%%"),
    })
    st.caption(
        "A useful risk rule has an elevated test churn rate and enough test customers "
        "to be interpretable. Compare its training and test rates; large gaps can mean "
        "the rule is unstable."
    )


def render_segments(data, scored):
    page_header("Unsupervised learning", "Customer segments",
                "Customers are grouped using services, contract, tenure and billing only. "
                "Churn is used afterwards to describe the groups, never to form them.")
    approach = st.radio(
        "Segmentation approach",
        ["Two-stage (recommended)", "Single-stage comparison"],
        horizontal=True,
        help="Two-stage separates customers without internet service, then groups internet customers. "
             "Single-stage runs K-Means, Ward and DBSCAN across all customers.",
    )
    two_stage = approach.startswith("Two-stage")
    with st.expander("Segmentation settings"):
        a, b, c = st.columns(3)
        k = a.slider("K-Means groups", 2, 6 if two_stage else 8, 3)
        collapse = b.checkbox("Merge repeated 'no service' categories", value=False,
                              help="Treat 'No internet service' and 'No phone service' as 'No' so the "
                                   "same absence is not counted several times in the distance.")
        extended = c.checkbox("Run k-selection and stability checks", value=False,
                              help="Slower: adds a k sweep plus seed and bootstrap stability.")
        if not two_stage:
            d, e = st.columns(2)
            eps = d.slider("DBSCAN eps", 0.5, 3.0, 2.0, 0.1)
            min_samples = e.slider("DBSCAN min_samples", 3, 30, 10)
    if two_stage:
        result = get_two_stage(data, k, collapse, extended)
        labels = result["labels"]
        profile = result["profile"][0]
        contracts = result["profile"][1]
        internet = result["profile"][2]
        map_coords, map_variance = get_map(data, collapse)
        current_name = f"Two-stage ({k + 1})"
        counts = st.columns(3)
        counts[0].metric("Total segments", profile.shape[0])
        counts[1].metric("Customers with internet", f"{result['internet_customers']:,}")
        counts[2].metric("Customers without internet", f"{len(data) - result['internet_customers']:,}")
        st.caption("The total includes one no-internet group plus the selected number of K-Means groups among internet customers.")
    else:
        result = get_clusters(data, k, eps, min_samples, collapse, extended)
        labels = result["kmeans_labels"]
        profile, contracts, internet = result["kmeans_profile"]
        map_coords, map_variance = result["pca_2d"], result["pca_variance"]
        current_name = "K-Means"
        counts = st.columns(4)
        counts[0].metric("K-Means clusters", len(np.unique(result["kmeans_labels"])))
        counts[1].metric("Ward clusters", len(np.unique(result["ward_labels"])))
        counts[2].metric("DBSCAN clusters", result["dbscan_cluster_count"])
        counts[3].metric("DBSCAN noise", f"{result['dbscan_noise_count']:,}")

    seg = scored[["Active", "Revenue at risk", "Monthly Charges"]].copy()
    seg["cluster"] = labels
    risk = seg.groupby("cluster")["Revenue at risk"].sum()
    overall = data[TARGET].mean() * 100
    cards = st.columns(min(len(profile), 4))
    for i, (cid, row) in enumerate(profile.iterrows()):
        lift = row["Churn_Lift"]
        label, cls = (("High churn", "pill-high") if lift >= 1.3
                      else ("Low churn", "pill-low") if lift <= 0.7 else ("Average churn", "pill-mid"))
        with cards[i % len(cards)]:
            card(f"#{cid} · {row['Segment']}", f"{row['Churn_Rate']:.0f}% churn",
                 f"{int(row['Customers']):,} customers ({row['Share_%']:.0f}%) · "
                 f"{money(risk.get(cid, 0))} monthly revenue at risk", label, cls)
        if (i + 1) % 4 == 0 and i + 1 < len(profile):
            cards = st.columns(min(len(profile) - i - 1, 4))
    st.write("")

    left, right = st.columns([1.5, 1])
    with left:
        colour_by = st.radio("Colour the map by", ["Segment", "Churn outcome"], horizontal=True)
        plot = pd.DataFrame(map_coords, columns=["Component 1", "Component 2"])
        plot["Segment"] = pd.Series(labels).map(profile["Segment"]).values
        plot["Churn outcome"] = data[TARGET].map({0: "Stayed", 1: "Churned"}).values
        fig = px.scatter(plot, x="Component 1", y="Component 2", color=colour_by, opacity=0.5,
                         color_discrete_sequence=PALETTE if colour_by == "Segment" else None,
                         color_discrete_map={"Stayed": TEAL, "Churned": CORAL} if colour_by != "Segment" else None,
                         title=f"Customer map (2 components explain {map_variance:.0%} of the variation)")
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(height=540, legend=dict(orientation="h", yanchor="top", y=-0.18,
                                                  xanchor="left", x=0))
        show(fig, key="sg_map")
    with right:
        bars = pd.DataFrame({"Segment": profile["Segment"].values,
                             "Churn_Rate": profile["Churn_Rate"].values})
        fig = px.bar(bars, x="Segment", y="Churn_Rate", text="Churn_Rate", title="Churn rate by segment")
        fig.update_traces(texttemplate="%{text:.0f}%", marker_color=CORAL, cliponaxis=False)
        fig.add_hline(y=overall, line_dash="dot", line_color=GREY,
                      annotation_text=f"All {overall:.0f}%", annotation_font_color=GREY)
        fig.update_layout(xaxis_title="", yaxis_title="Churn rate (%)", height=540)
        show(fig, key="sg_bars")
    st.caption("The map compresses many features into two dimensions, so overlap is expected. "
               "Read groups as practical summaries; the checks below show how strongly separated they are.")

    tab_names = (["Profiles", "Detailed profiles", "Churn reasons", "How real are these segments?",
                  "Risk segments (supervised)", "Choosing k"] if two_stage else
                 ["Profiles", "Detailed profiles", "Churn reasons", "Method comparison",
                  "Choosing k", "Dendrogram", "DBSCAN", "How real are these segments?",
                  "Risk segments (supervised)"])
    tabs = st.tabs(tab_names)
    tab = dict(zip(tab_names, tabs))
    with tab["Profiles"]:
        table(format_profile(profile))
        a, b = st.columns(2)
        with a:
            st.markdown("**Contract mix (%)**")
            table(contracts.round(1))
        with b:
            st.markdown("**Internet service mix (%)**")
            table(internet.round(1))
    with tab["Detailed profiles"]:
        if two_stage:
            method, detail_labels, detail_profile = "Two-stage", labels, profile
        else:
            method = st.selectbox("Clustering method", ["K-Means", "Agglomerative Ward", "DBSCAN"],
                                  key="detail_method")
            label_key, profile_key = {
                "K-Means": ("kmeans_labels", "kmeans_profile"),
                "Agglomerative Ward": ("ward_labels", "ward_profile"),
                "DBSCAN": ("dbscan_labels", "dbscan_profile"),
            }[method]
            detail_labels = result[label_key]
            detail_profile = result[profile_key][0]
        cluster_ids = sorted(detail_profile.index.tolist())
        selected_cluster = st.selectbox(
            "Select cluster",
            cluster_ids,
            format_func=lambda cid: (
                f"Noise (-1) · {detail_profile.loc[cid, 'Customers']:,} customers"
                if cid == -1 else
                f"Cluster {cid} · {detail_profile.loc[cid, 'Customers']:,} customers"
                f" · {detail_profile.loc[cid, 'Segment']}"
            ),
            key="detail_cluster",
        )
        st.markdown("**Cluster summary**")
        table(format_profile(detail_profile.loc[[selected_cluster]]))

        feature_col, numeric_col = st.columns(2)
        with feature_col:
            categorical_feature = st.selectbox(
                "Categorical feature mix",
                CATEGORICAL_FEATURES,
                key="detail_feature",
            )
            mix = cluster_feature_mix(
                data, detail_labels, selected_cluster, categorical_feature
            )
            st.caption(
                f"Counts and percentages are within this {method} group. "
                "Choose another field to inspect its composition."
            )
            table(
                mix.style.format({"Share within cluster (%)": "{:.1f}%"}),
                hide_index=True,
            )
        with numeric_col:
            numeric_feature = st.selectbox(
                "Compare numeric feature distributions",
                NUMERIC_FEATURES,
                key="detail_numeric",
            )
            numeric_plot = data[[numeric_feature]].copy()
            numeric_plot["Segment"] = pd.Series(detail_labels).map(
                detail_profile["Segment"]
            ).values
            fig = px.box(
                numeric_plot, x="Segment", y=numeric_feature, color="Segment",
                points=False, title=f"{numeric_feature} across {method} groups",
            )
            fig.update_layout(
                xaxis_title="", yaxis_title=numeric_feature,
                showlegend=False, height=390,
            )
            show(fig, key=f"detail_num_{method}_{numeric_feature}")
        st.caption(
            "Cluster IDs are arbitrary labels. DBSCAN label -1 is its noise/unassigned group."
        )
    with tab["Churn reasons"]:
        st.write(
            "These are the reasons churned customers gave after leaving. They are "
            "summarized within clusters after clustering; they were not used to form "
            "clusters or train the churn model."
        )
        if two_stage:
            reason_method, reason_labels, reason_profile = "Two-stage", labels, profile
        else:
            reason_method = st.selectbox("Cluster method for reason analysis",
                                         ["K-Means", "Agglomerative Ward", "DBSCAN"],
                                         key="reason_method")
            reason_label_key, reason_profile_key = {
                "K-Means": ("kmeans_labels", "kmeans_profile"),
                "Agglomerative Ward": ("ward_labels", "ward_profile"),
                "DBSCAN": ("dbscan_labels", "dbscan_profile"),
            }[reason_method]
            reason_labels = result[reason_label_key]
            reason_profile = result[reason_profile_key][0]
        reason_clusters = sorted(reason_profile.index.tolist())
        reason_cluster = st.selectbox(
            "Cluster to inspect",
            reason_clusters,
            format_func=lambda cid: (
                f"Noise (-1)" if cid == -1 else
                f"Cluster {cid} · {reason_profile.loc[cid, 'Segment']}"
            ),
            key="reason_cluster",
        )
        if "Churn Reason" not in data.columns:
            st.info("This workbook does not contain a Churn Reason column.")
        else:
            reason_rows = cluster_reason_mix(data, reason_labels, reason_cluster)
            n_churners = int(
                ((np.asarray(reason_labels) == reason_cluster) & data[TARGET].eq(1).to_numpy()).sum()
            )
            st.metric("Churned customers in this group", f"{n_churners:,}")
            if reason_rows.empty:
                st.info("There are no churned customers in this group to summarize.")
            else:
                left, right = st.columns([1.2, 1])
                with left:
                    table(
                        reason_rows.style.format({
                            "Share of cluster churners (%)": "{:.1f}%"
                        }),
                        hide_index=True,
                    )
                with right:
                    plot_rows = reason_rows.sort_values("Churned customers")
                    fig = px.bar(
                        plot_rows, x="Churned customers", y="Reason", orientation="h",
                        color="Churned customers", color_continuous_scale=[TEAL, CORAL],
                        title="Reported reasons among churners in this group",
                    )
                    fig.update_layout(
                        xaxis_title="Churned customers", yaxis_title="",
                        coloraxis_showscale=False, height=430,
                    )
                    show(fig, key=f"reason_{reason_method}_{reason_cluster}")
                st.caption(
                    "Percentages use churned customers in the selected group as the "
                    "denominator. Self-reported reasons describe customers' answers; "
                    "they do not establish causes."
                )
    if not two_stage:
      with tab["Method comparison"]:
        c1, c2 = st.columns([2, 1])
        with c1: table(result["metrics"].style.format("{:.3f}"))
        c2.metric("K-Means vs Ward agreement (ARI)", f"{result['ari']:.3f}")
        st.caption("Silhouette is sampled from up to 2,000 customers; Davies–Bouldin and "
                   "Calinski–Harabasz use all customers. Higher silhouette/Calinski–Harabasz "
                   "and lower Davies–Bouldin are generally preferred. ARI of 1 means identical assignments.")
        st.markdown("**How K-Means and Ward overlap** (Ward numbers matched to K-Means)")
        table(result["kmeans_vs_ward"])
        st.markdown("**Ward profiles**")
        table(format_profile(result["ward_profile"][0]))
    ks = result["k_selection"] if two_stage else result["k_selection"]
    stable = result["stability"] if extended else None
    with tab["Choosing k"]:
        render_choosing_k(ks, stable, two_stage)
    if not two_stage:
      with tab["Dendrogram"]:
        leaves = st.slider("Visible merged groups", 15, 60, 30, key="leaves")
        tree = dendrogram_data(result["ward_linkage"], leaves)
        fig = go.Figure()
        for xs, ys in zip(tree["icoord"], tree["dcoord"]):
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=NAVY, width=1),
                                     hoverinfo="skip", showlegend=False))
        fig.update_layout(title="Ward dendrogram (truncated)", xaxis_title="Merged customer groups",
                          yaxis_title="Ward distance", height=480)
        show(fig, key="sg_dendro")
        st.caption("Tall vertical jumps mark groups that are relatively different from each other.")
      with tab["DBSCAN"]:
        a, b, c = st.columns(3)
        a.metric("Clusters found", result["dbscan_cluster_count"])
        b.metric("Noise customers", f"{result['dbscan_noise_count']:,}")
        c.metric("Coverage", f"{result['dbscan_coverage_percent']:.0f}%")
        table(format_profile(result["dbscan_profile"][0]))
        if result["dbscan_metrics"]:
            st.markdown("**Internal scores (noise customers excluded)**")
            table(pd.DataFrame([result["dbscan_metrics"]], index=["Non-noise"]).style.format("{:.3f}"))
        note("Noise means the customer is not in a dense group at these settings. Because one-hot data "
             "has stepped distances, DBSCAN labels a large share of customers as noise here, and its "
             "scores cover only the rest, so they are not directly comparable with K-Means or Ward.")
    with tab["How real are these segments?"]:
        render_how_real(data, labels, profile["Segment"].to_dict(), current_name)
    with tab["Risk segments (supervised)"]:
        render_risk_tree(data)


def render_actions(data, scored):
    page_header("Action center", "Who to contact first",
                "Active customers ranked by estimated churn probability. Every customer is scored by "
                "a model that never saw them, so the ranking is honest.")
    f1, f2, f3, f4 = st.columns([1.3, 1.1, 1.2, 1])
    segments = f1.multiselect("Segment", sorted(scored["Segment"].unique()), placeholder="All")
    contracts = f2.multiselect("Contract", sorted(scored["Contract"].unique()), placeholder="All")
    min_prob = f3.slider("Minimum churn probability", 0, 90, 30, 5, format="%d%%") / 100
    active_only = f4.toggle("Active customers only", value=True,
                            help="Customers who already left cannot be saved, so they are hidden by default.")
    view = scored.copy()
    if active_only:
        view = view[view["Active"]]
    if segments:
        view = view[view["Segment"].isin(segments)]
    if contracts:
        view = view[view["Contract"].isin(contracts)]
    view = view[view["Churn probability"] >= min_prob].sort_values("Churn probability", ascending=False)

    cols = st.columns(4)
    kpi(cols[0], "Customers in list", f"{len(view):,}")
    kpi(cols[1], "Expected churners", f"{view['Churn probability'].sum():,.0f}", "sum of probabilities")
    kpi(cols[2], "Revenue at risk / month", money(view["Revenue at risk"].sum()))
    kpi(cols[3], "Flagged by the model", f"{int(view['Flagged'].sum()):,}", "raw score ≥ 0.50")
    if view.empty:
        st.info("No customers match these filters.")
        return
    id_col = "CustomerID" if "CustomerID" in view.columns else None
    out = pd.DataFrame({
        **({"Customer": view[id_col].values} if id_col else {}),
        "Segment": view["Segment"].values,
        "Risk": view["Risk"].values,
        "Churn probability": (view["Churn probability"] * 100).round(1).values,
        "Contract": view["Contract"].values,
        "Internet": view["Internet Service"].values,
        "Tenure (mo)": view["Tenure Months"].values,
        "Monthly charge": view["Monthly Charges"].round(2).values,
        "Monthly revenue at risk": view["Revenue at risk"].round(2).values,
    })
    shown = out.head(300)
    table(shown, hide_index=True, column_config={
        "Churn probability": st.column_config.ProgressColumn(
            "Churn probability", min_value=0, max_value=100, format="%.0f%%"),
        "Monthly charge": st.column_config.NumberColumn(format="$%.2f"),
        "Monthly revenue at risk": st.column_config.NumberColumn(format="$%.2f"),
    }, height=480)
    st.caption(f"Showing the top {len(shown):,} of {len(out):,}. The download contains the full list.")
    st.caption(
        "Here, active means Churn Value = 0 in this dataset snapshot. The list is a "
        "historical model-based ranking, not a guarantee those customers will churn."
    )
    st.download_button("Download this list (CSV)", out.to_csv(index=False).encode("utf-8"),
                       "churn_action_list.csv", "text/csv", type="primary")
    note("Risk: <b>High</b> = flagged by the model and calibrated probability ≥ 50%; <b>Medium</b> = flagged "
         "(raw score ≥ 0.50) with a lower probability; <b>Low</b> = not flagged. Revenue at risk = "
         "probability × monthly charge.")


def render_conclusions(data, scored):
    page_header("Interpretation", "What to take away",
                "Findings from this dataset, the choices made, and what to be careful about.")
    result = get_models(data)
    m = metrics_at_threshold(result["y_test"], result["test_probabilities"], FLAG)
    tn, fp, fn, tp = m["Confusion matrix"].ravel()
    cal = result["calibration"]
    active = scored[scored["Active"]]
    rate = data[TARGET].mean()
    st.subheader("Key findings")
    insight(f"The dataset has <b>{len(data):,}</b> customers with an overall churn rate of <b>{rate:.1%}</b>. "
            "Contract type, tenure and internet service separate high and low churn the most.")
    insight(f"XGBoost at threshold 0.50 finds <b>{m['Churn recall']:.0%}</b> of churners "
            f"(<b>{fn}</b> missed) with precision <b>{m['Churn precision']:.0%}</b> and ROC AUC "
            f"<b>{m['ROC AUC']:.3f}</b> on held-out customers.")
    insight(f"On this test split, calibration changes Brier from <b>{cal['Brier raw']:.3f}</b> "
            f"to <b>{cal['Brier calibrated']:.3f}</b> and ECE from <b>{cal['ECE raw']:.3f}</b> "
            f"to <b>{cal['ECE calibrated']:.3f}</b>. This supports better probability estimates, "
            "but they still need validation on later customer data.")
    insight(f"Among customers who have not yet churned, <b>{int(active['Flagged'].sum()):,}</b> are flagged, "
            f"carrying about <b>{money(active['Revenue at risk'].sum())}</b> of monthly revenue at risk.")

    st.subheader("Choices and what was tried")
    st.markdown(
        "| Topic | Decision |\n|---|---|\n"
        "| **Model** | XGBoost with class weighting; churn recall is a priority. |\n"
        "| **Decision rule** | The reference flag is raw score ≥ 0.50; isotonic calibration is used separately for probability estimates. |\n"
        "| **Segmentation** | Two-stage is the default: no-internet customers are separated, then K-Means groups internet customers; single-stage K-Means, Ward and DBSCAN remain available for comparison. |\n"
        "| **Distance sensitivity** | Collapsing repeated 'no service' categories is available as a sensitivity check; it is not the default because it lowered K-Means internal scores on this dataset. |"
    )

    st.subheader("Limitations")
    st.markdown(
        "- Results describe this dataset and may not transfer to another company, market or period.\n"
        "- One held-out split can score differently on a new sample; cross-validation shows the spread.\n"
        "- Model explanations and segments are associations, not causes; test retention actions before "
        "claiming impact.\n"
        "- The dataset is a snapshot: the action list defaults to customers marked retained in this file; "
        "its toggle can include churned customers for historical comparison.\n"
        "- HDBSCAN and silhouette checks suggest the no-internet group is the clearest structural split; "
        "internet-customer groups overlap and are better read as gradients than sharply separated types.\n"
        "- Cluster quality is modest and one-hot distances are stepped, so treat segments as practical "
        "groupings and DBSCAN settings as sensitive. The supervised risk tree is a separate, label-trained grouping.\n"
        "- Churn reasons are collected after customers leave and are never used for prediction.")
    st.subheader("Next steps")
    st.write("Validate on later-period customers, set the threshold from real outreach costs on the "
             "Prediction page, and run a small retention test on the highest-risk segment.")


# --------------------------------------------------------------------------- #
# App shell
# --------------------------------------------------------------------------- #
def main():
    with st.sidebar:
        st.markdown('<div class="brand">Churn <span>Intelligence</span></div>'
                    '<div class="brand-sub">Prediction · Segments · Action</div>', unsafe_allow_html=True)
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
        st.divider()
        uploaded = st.file_uploader("Use a different workbook", type=["xlsx"])
    try:
        data = get_data(uploaded.getvalue() if uploaded else None, uploaded.name if uploaded else "")
    except DatasetError as exc:
        st.error(str(exc))
        if "openpyxl" in str(exc).lower():
            st.info(
                "Install the project requirements in the same Python environment used "
                "to start Streamlit, then restart the app."
            )
        else:
            st.info(
                "Check that the workbook has the required churn columns, or upload a "
                "compatible .xlsx file."
            )
        st.stop()
        return
    except FileNotFoundError:
        st.error("The default Telco_customer_churn.xlsx workbook was not found.")
        st.info("Place it one level above the streamlit_app folder, or upload a compatible .xlsx file in the sidebar.")
        st.stop()
        return
    st.sidebar.caption(f"{len(data):,} customers loaded")
    needs_scores = page in ("Overview", "Action center", "Segments", "Conclusions")
    scored = get_scored(data) if needs_scores else None
    if page == "Overview":
        render_overview(data, scored)
    elif page == "Explore":
        render_explore(data)
    elif page == "Prediction":
        render_prediction(data, None)
    elif page == "Segments":
        render_segments(data, scored)
    elif page == "Action center":
        render_actions(data, scored)
    else:
        render_conclusions(data, scored)


if __name__ == "__main__":
    main()
