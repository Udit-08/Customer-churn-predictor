import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    normalized_mutual_info_score, silhouette_samples, silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, FEATURES, TARGET

SEED = 42
ADDON_SERVICES = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies",
]
# Silhouette is estimated from a reproducible sample to keep interactive use fast.
SILHOUETTE_SAMPLE_LIMIT = 2_000


# --------------------------------------------------------------------------- #
# Feature matrix
# --------------------------------------------------------------------------- #
def collapse_redundant_categories(data):
    """Merge categories that only repeat another column.

    "No internet service" in the six add-on columns says the same thing as
    Internet Service = "No", and "No phone service" in Multiple Lines repeats
    Phone Service = "No". Left as separate one-hot columns they count the same
    fact up to seven times in the distance calculation.
    """
    out = data.copy()
    for column in ADDON_SERVICES:
        out[column] = out[column].replace("No internet service", "No")
    out["Multiple Lines"] = out["Multiple Lines"].replace("No phone service", "No")
    return out


def prepare_matrix(data, collapse_redundant=False):
    work = collapse_redundant_categories(data) if collapse_redundant else data
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformer = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    matrix = transformer.fit_transform(work[FEATURES])
    # Ward linkage requires dense input. The Telco feature matrix is small enough
    # to densify here (about a few hundred thousand values for this workbook).
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def _internal_metrics(matrix, labels):
    """Silhouette, Davies-Bouldin and Calinski-Harabasz on ALL rows.

    Calinski-Harabasz grows with the number of rows, so it is never computed on
    a sample (a 2,000-row sample reports about 627 where the full data gives
    about 2,243). Only silhouette is sampled, and only above a size limit.
    """
    if len(np.unique(labels)) < 2:
        return {"Silhouette": np.nan, "Davies-Bouldin": np.nan,
                "Calinski-Harabasz": np.nan}
    sil_matrix, sil_labels = matrix, labels
    if len(labels) > SILHOUETTE_SAMPLE_LIMIT:
        rng = np.random.RandomState(SEED)
        chosen = rng.choice(len(labels), SILHOUETTE_SAMPLE_LIMIT, replace=False)
        sil_matrix, sil_labels = matrix[chosen], labels[chosen]
    silhouette = (silhouette_score(sil_matrix, sil_labels)
                  if len(np.unique(sil_labels)) > 1 else np.nan)
    return {
        "Silhouette": silhouette,
        "Davies-Bouldin": davies_bouldin_score(matrix, labels),
        "Calinski-Harabasz": calinski_harabasz_score(matrix, labels),
    }


def k_selection(matrix, ks=range(2, 9)):
    """Score K-Means for several k so the choice of k is shown, not assumed."""
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(matrix)
        rows.append({"k": k, **_internal_metrics(matrix, model.labels_),
                     "Inertia": model.inertia_})
    return pd.DataFrame(rows).set_index("k")


def stability(matrix, k=3, n_runs=10):
    """How much K-Means changes with a different seed or a resampled dataset.

    ARI of 1.0 means identical groupings.
    """
    reference = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(matrix)
    seed_ari = [
        adjusted_rand_score(
            reference.labels_,
            KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(matrix),
        )
        for seed in range(n_runs)
    ]
    rng = np.random.RandomState(SEED)
    boot_ari = []
    for run in range(n_runs):
        rows = rng.choice(len(matrix), len(matrix), replace=True)
        boot = KMeans(n_clusters=k, n_init=10, random_state=run).fit(matrix[rows])
        boot_ari.append(adjusted_rand_score(reference.labels_, boot.predict(matrix)))
    return {
        "seed_ari_mean": float(np.mean(seed_ari)),
        "seed_ari_min": float(np.min(seed_ari)),
        "bootstrap_ari_mean": float(np.mean(boot_ari)),
        "bootstrap_ari_min": float(np.min(boot_ari)),
        "runs": n_runs,
    }


# --------------------------------------------------------------------------- #
# Labels and profiles
# --------------------------------------------------------------------------- #
def align_labels(reference, labels):
    """Renumber `labels` so each cluster takes the id of the reference cluster
    it overlaps most (cluster ids are arbitrary, so 0 in one method is not 0 in
    another). Uses an optimal one-to-one matching."""
    ref_ids = np.unique(reference)
    lab_ids = np.unique(labels)
    overlap = np.array([[np.sum((labels == lab) & (reference == ref))
                         for ref in ref_ids] for lab in lab_ids])
    rows, cols = linear_sum_assignment(-overlap)
    mapping = {lab_ids[r]: ref_ids[c] for r, c in zip(rows, cols)}
    next_id = max(ref_ids.max(), lab_ids.max()) + 1
    for lab in lab_ids:
        if lab not in mapping:
            mapping[lab] = next_id
            next_id += 1
    return np.array([mapping[lab] for lab in labels])


def _segment_names(summary, contracts, internet, tenure_cutoffs):
    names = {}
    low, high = tenure_cutoffs
    for cluster_id in summary.index:
        if cluster_id == -1:
            names[cluster_id] = "Noise / unassigned"
            continue
        # Only name a contract type when it is the majority; otherwise say "mixed".
        contract = (contracts.loc[cluster_id].idxmax()
                    if contracts.loc[cluster_id].max() >= 50 else "Mixed contracts")
        top_internet = internet.loc[cluster_id].idxmax()
        if internet.loc[cluster_id].max() < 60:
            internet_label = "mixed internet"
        elif top_internet == "No":
            internet_label = "no internet"
        else:
            internet_label = f"mostly {top_internet.lower()}"
        tenure = summary.loc[cluster_id, "Median_Tenure"]
        tenure_label = ("short" if tenure <= low
                        else "long" if tenure >= high else "mid")
        names[cluster_id] = f"{contract} · {internet_label} · {tenure_label} tenure"
    # Keep names unique so charts and tables never merge two segments.
    seen = {}
    for cluster_id, name in names.items():
        seen.setdefault(name, []).append(cluster_id)
    for name, ids in seen.items():
        if len(ids) > 1:
            for cluster_id in ids:
                names[cluster_id] = f"{name} (#{cluster_id})"
    return names


def _profile(data, labels, label_name):
    work = data.copy()
    work[label_name] = labels
    overall_rate = work[TARGET].mean() * 100
    total_churners = work[TARGET].sum()
    group = work.groupby(label_name, dropna=False)

    summary = group.agg(
        Customers=(TARGET, "size"),
        Churn_Rate=(TARGET, "mean"),
        Median_Tenure=("Tenure Months", "median"),
        Avg_Monthly_Charges=("Monthly Charges", "mean"),
        Avg_Total_Charges=("Total Charges", "mean"),
    )
    summary["Churn_Rate"] *= 100
    # Extra columns go at the end so existing column references keep working.
    summary["Share_%"] = summary["Customers"] / len(work) * 100
    summary["Churn_Lift"] = summary["Churn_Rate"] / overall_rate
    summary["Churners_Captured_%"] = group[TARGET].sum() / total_churners * 100
    summary["Dependents_Yes_%"] = group["Dependents"].apply(lambda s: (s == "Yes").mean() * 100)
    summary["Senior_Yes_%"] = group["Senior Citizen"].apply(lambda s: (s == "Yes").mean() * 100)
    summary["Electronic_Check_%"] = group["Payment Method"].apply(
        lambda s: (s == "Electronic check").mean() * 100)
    addon_count = sum((work[c] == "Yes").astype(int) for c in ADDON_SERVICES)
    summary["Avg_Addon_Services"] = addon_count.groupby(work[label_name]).mean()

    contracts = pd.crosstab(work[label_name], work["Contract"], normalize="index") * 100
    internet = pd.crosstab(work[label_name], work["Internet Service"], normalize="index") * 100
    cutoffs = tuple(work["Tenure Months"].quantile([1 / 3, 2 / 3]))
    summary["Segment"] = pd.Series(_segment_names(summary, contracts, internet, cutoffs))
    return summary, contracts, internet


def _linkage_matrix(model):
    children = model.children_
    distances = model.distances_
    counts = np.zeros(children.shape[0])
    n = len(model.labels_)
    for i, merge in enumerate(children):
        count = 0
        for child in merge:
            count += 1 if child < n else counts[child - n]
        counts[i] = count
    return np.column_stack([children, distances, counts]).astype(float)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def fit_clustering(data, k=3, eps=2.0, min_samples=10, collapse_redundant=False,
                   include_k_sweep=False, include_stability=False):
    """Fit K-Means, Ward and DBSCAN and profile the clusters.

    collapse_redundant=False reproduces the original notebook features.
    Churn is only used to PROFILE clusters, never to form them.
    """
    matrix = prepare_matrix(data, collapse_redundant=collapse_redundant)
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    km_labels = km.fit_predict(matrix)
    ward = AgglomerativeClustering(
        n_clusters=k, linkage="ward", compute_full_tree=True, compute_distances=True
    )
    ward_raw = ward.fit_predict(matrix)
    # Same ids as K-Means where the groups overlap, so tables line up.
    ward_labels = align_labels(km_labels, ward_raw)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    db_labels = db.fit_predict(matrix)

    metrics = []
    for name, labels in [("K-Means", km_labels), ("Agglomerative Ward", ward_labels)]:
        metrics.append({"Method": name, **_internal_metrics(matrix, labels)})
    ari = adjusted_rand_score(km_labels, ward_labels)
    km_profile = _profile(data, km_labels, "Cluster")
    ward_profile = _profile(data, ward_labels, "Agg_Cluster")
    db_profile = _profile(data, db_labels, "DBSCAN_Cluster")

    # DBSCAN scores cover only the points it did not call noise, so they are
    # not directly comparable with K-Means/Ward (which label everyone).
    non_noise = db_labels != -1
    db_metrics = (_internal_metrics(matrix[non_noise], db_labels[non_noise])
                  if non_noise.sum() > 1 else {})
    noise_count = int((db_labels == -1).sum())
    # 2-D map of the feature space (for plotting only; clustering used all features).
    pca = PCA(n_components=2, random_state=SEED)
    pca_2d = pca.fit_transform(matrix)
    pca_variance = float(pca.explained_variance_ratio_.sum())
    return {
        "metrics": pd.DataFrame(metrics).set_index("Method"),
        "ari": ari,
        "kmeans_labels": km_labels,
        "ward_labels": ward_labels,
        "ward_labels_raw": ward_raw,
        "dbscan_labels": db_labels,
        "kmeans_vs_ward": pd.crosstab(
            pd.Series(km_labels, name="K-Means"),
            pd.Series(ward_labels, name="Ward"), margins=True),
        "kmeans_profile": km_profile,
        "ward_profile": ward_profile,
        "dbscan_profile": db_profile,
        "ward_linkage": _linkage_matrix(ward),
        "dbscan_noise_count": noise_count,
        "dbscan_noise_percent": noise_count / len(db_labels) * 100,
        "dbscan_coverage_percent": 100 - noise_count / len(db_labels) * 100,
        "dbscan_cluster_count": len(set(db_labels) - {-1}),
        "dbscan_metrics": db_metrics,
        "k_selection": k_selection(matrix) if include_k_sweep else None,
        "stability": stability(matrix, k) if include_stability else None,
        "collapse_redundant": collapse_redundant,
        "pca_2d": pca_2d,
        "pca_variance": pca_variance,
    }


def dendrogram_data(linkage_matrix, leaf_count=30):
    return dendrogram(
        linkage_matrix, truncate_mode="lastp", p=leaf_count,
        show_leaf_counts=True, no_plot=True,
    )


# --------------------------------------------------------------------------- #
# Two-stage segmentation and structure checks
# --------------------------------------------------------------------------- #
RULE_NAMES = [
    "New month-to-month", "Older month-to-month",
    "Committed (1-2 year)", "No internet",
]
RULE_TEXT = {
    "New month-to-month": "Internet, month-to-month, tenure 12 months or less",
    "Older month-to-month": "Internet, month-to-month, tenure over 12 months",
    "Committed (1-2 year)": "Internet, one- or two-year contract",
    "No internet": "No internet service",
}


def fit_two_stage(
    data, k=3, collapse_redundant=False,
    include_k_sweep=False, include_stability=False,
):
    """Set aside no-internet customers, then cluster internet customers.

    Churn is not part of the feature matrix or the cluster numbering. The
    no-internet group receives ID k; internet group IDs come from K-Means.
    """
    has_internet = data["Internet Service"].ne("No").to_numpy()
    internet_data = data.loc[has_internet]
    matrix = prepare_matrix(internet_data, collapse_redundant=collapse_redundant)
    model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(matrix)
    labels = np.full(len(data), k, dtype=int)
    labels[has_internet] = model.labels_
    return {
        "labels": labels,
        "profile": _profile(data, labels, "Segment_ID"),
        "internet_customers": int(has_internet.sum()),
        "internet_scores": _internal_metrics(matrix, model.labels_),
        "k_selection": k_selection(matrix, range(2, 7)) if include_k_sweep else None,
        "stability": stability(matrix, k) if include_stability else None,
    }


def pca_map(data, collapse_redundant=False):
    """Return a two-dimensional display projection and its explained variance."""
    pca = PCA(n_components=2, random_state=SEED)
    coords = pca.fit_transform(
        prepare_matrix(data, collapse_redundant=collapse_redundant)
    )
    return coords, float(pca.explained_variance_ratio_.sum())


def natural_structure(data, sizes=(100, 200, 400)):
    """Run sklearn HDBSCAN on a 10-component PCA projection.

    Returns None if HDBSCAN is unavailable in the installed scikit-learn.
    """
    try:
        from sklearn.cluster import HDBSCAN
    except ImportError:
        return None
    reduced = PCA(n_components=10, random_state=SEED).fit_transform(
        prepare_matrix(data)
    )
    rows = []
    for size in sizes:
        labels = HDBSCAN(min_cluster_size=size).fit_predict(reduced)
        found = [cluster for cluster in np.unique(labels) if cluster != -1]
        rows.append({
            "Min cluster size": size,
            "Clusters found": len(found),
            "Noise (%)": float(np.mean(labels == -1) * 100),
            "Cluster sizes": ", ".join(
                f"{int((labels == cluster).sum()):,}" for cluster in found
            ) or "none",
        })
    return pd.DataFrame(rows).set_index("Min cluster size")


def silhouette_by_label(data, labels):
    """Average silhouette per group in the full transformed feature space."""
    scores = silhouette_samples(prepare_matrix(data), labels)
    return pd.Series(scores).groupby(np.asarray(labels)).mean()


def rule_labels(data):
    """Simple, pre-defined rules; churn outcomes do not define the groups."""
    month_to_month = data["Contract"].eq("Month-to-month")
    new_customer = data["Tenure Months"].le(12)
    return np.select(
        [
            data["Internet Service"].eq("No"),
            month_to_month & new_customer,
            month_to_month,
        ],
        [3, 0, 1],
        default=2,
    )


def rule_table(data):
    labels = rule_labels(data)
    frame = pd.DataFrame({"label": labels, TARGET: data[TARGET].to_numpy()})
    grouped = frame.groupby("label")[TARGET].agg(["size", "sum", "mean"])
    result = pd.DataFrame({
        "Segment": [RULE_NAMES[index] for index in grouped.index],
        "Rule": [RULE_TEXT[RULE_NAMES[index]] for index in grouped.index],
        "Customers": grouped["size"].to_numpy(),
        "Churn rate": grouped["mean"].to_numpy() * 100,
        "Share of churners": grouped["sum"].to_numpy() / frame[TARGET].sum() * 100,
    })
    return result.sort_values("Churn rate", ascending=False).reset_index(drop=True)


def compare_segmentations(data, labelings):
    """Compare group sizes and post-hoc churn separation across approaches."""
    y = data[TARGET].to_numpy()
    total_churners = y.sum()
    rows = []
    for name, labels in labelings.items():
        labels = np.asarray(labels)
        grouped = (
            pd.DataFrame({"label": labels, "churn": y})
            .groupby("label")["churn"].agg(["size", "sum", "mean"])
        )
        top = grouped["mean"].idxmax()
        rows.append({
            "Approach": name,
            "Segments": len(grouped),
            "Highest churn (%)": grouped["mean"].max() * 100,
            "Lowest churn (%)": grouped["mean"].min() * 100,
            "Customers in top segment": int(grouped.loc[top, "size"]),
            "Churners in top segment (%)": grouped.loc[top, "sum"] / total_churners * 100,
            "Link to churn (NMI)": normalized_mutual_info_score(labels, y),
        })
    return pd.DataFrame(rows).set_index("Approach")
