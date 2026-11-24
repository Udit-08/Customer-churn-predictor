# Customer Churn Intelligence

A Streamlit application for exploring the Telco customer churn project through
exploratory analysis, supervised churn prediction, and unsupervised customer
segmentation. The notebooks and source workbook in the parent project folder are
reference materials and are not modified by the app.

## Features

- Filterable exploratory charts, two-way churn heatmap, churn reasons, and
  tenure/billing comparisons.
- XGBoost evaluation on a reproducible stratified 80/20 split, five-fold
  training-only cross-validation, threshold/cost exploration, probability
  calibration, gain curve, and permutation importance.
- Customer prediction form with calibrated probability and per-customer
  explanation.
- Survival analysis with Kaplan–Meier retention curves, a Cox proportional-
  hazards model, tenure-band event rates, holdout discrimination/calibration
  checks, and a retention curve for the customer entered in the prediction form.
- An action center that ranks existing retained customers with out-of-fold
  estimates and can export the filtered contact list as CSV.
- Two-stage segmentation by default: separate customers without internet service,
  then run K-Means on internet customers. Single-stage K-Means, Agglomerative
  Ward, and DBSCAN across all customers remain available for comparison.
- Cluster profiles with churn lift and captured-churn share, internal metrics,
  K-Means/Ward Adjusted Rand Index, a PCA map, and a truncated Ward dendrogram.
- Explicit cluster counts and detailed categorical/numeric profiles for each
  method, plus post-hoc Churn Reason breakdowns among churned customers in each
  selected cluster.
- A segment-structure check using HDBSCAN, silhouette by displayed segment,
  and comparison with simple customer rules. HDBSCAN is available in
  scikit-learn 1.3 and newer; the project requirements install a newer version.
- A separate shallow decision tree that creates supervised risk groups and
  reports churn rates on held-out test customers. It is clearly distinguished
  from unsupervised customer segmentation.
- Optional K-Means cluster-count and stability checks, plus an option to collapse
  repeated no-service categories in distance calculations.
- Data checks and a clear error when required workbook columns are missing.

## Data and leakage handling

By default the app loads Telco_customer_churn.xlsx from the project folder
one level above this directory. Alternatively, upload a compatible .xlsx
workbook from the sidebar.

The prediction and clustering inputs follow the notebook feature set:

- Categorical: Gender, Senior Citizen, Partner, Dependents, Phone Service,
  Multiple Lines, Internet Service, Online Security, Online Backup, Device
  Protection, Tech Support, Streaming TV, Streaming Movies, Contract,
  Paperless Billing, Payment Method.
- Numeric: Tenure Months, Monthly Charges, Total Charges.

Churn Value is the prediction target and is used only after clustering for
segment profiling. Churn Label, Churn Reason, Churn Score, CLTV, customer IDs,
and location fields are not model inputs. Blank Total Charges for zero-tenure
customers are set to zero, matching the notebook's treatment; any other numeric
missing values are imputed from training data within each model pipeline.

## Setup and run

From a terminal:

    cd streamlit_app
    python -m venv .venv

Activate the environment:

Windows PowerShell:

    .venv\Scripts\Activate.ps1

Windows Command Prompt:

    .venv\Scripts\activate.bat

Install dependencies and launch:

    python -m pip install -r requirements.txt
    streamlit run app.py

The application opens in your browser. Model fitting, out-of-fold scoring, and
hierarchical clustering can take a little time on their first use; the results
are cached for later interactions.

## Reading the results

Churn precision measures how often a churn flag is correct; churn recall measures
how many actual churners are found. ROC AUC measures ranking quality across
classification thresholds. The app reports these on a held-out test set and on
five stratified folds of training data. The threshold/cost page explores thresholds
using out-of-fold training scores. Its outreach success, retention value, and
contact cost fields are user-supplied assumptions, not measured facts.

Class weighting can make raw XGBoost scores run high. The app fits isotonic
calibration from training out-of-fold predictions and reports Brier score and
calibration error on the untouched test set. It also shows raw-score decisions
at the reference 0.50 threshold. The customer estimate and action center use
calibrated probabilities; action-center scores are out-of-fold estimates, not
guarantees of future behavior. The action center treats Churn Value = 0 as still
retained in this snapshot. Revenue at risk is a simple monthly estimate:
calibrated churn probability multiplied by monthly charges.

The survival page uses tenure as observed duration and Churn Value as the event
indicator; customers who have not churned are treated as right-censored. Tenure
and Total Charges are excluded from Cox predictors. The holdout encoder and
monthly-charge scaling are fitted on training rows only. Personalized retention
curves use the final model refitted on all supplied rows and are estimates under
the Cox model assumptions. The proportional-hazards assumption is not tested by
the app.

For clustering, higher Silhouette and Calinski-Harabasz scores are generally
preferred, while lower Davies-Bouldin is generally preferred. DBSCAN label -1
denotes noise under the chosen density settings, not an ordinary cluster.
The app estimates Silhouette from a reproducible sample of up to 2,000 customers;
Davies-Bouldin and Calinski-Harabasz use all customers (DBSCAN scores exclude
noise). Optional stability and cluster-count checks take longer to compute. PCA
is only used to draw the two-dimensional map; clustering is performed on the
full transformed feature matrix. The HDBSCAN structural check uses a 10-component
projection. On this dataset it finds the no-internet split consistently; the
internet-customer groups overlap more and should be interpreted as gradients.
The risk tree is trained using churn labels and is for prioritization, not
unsupervised customer typing.

Cluster patterns, model explanations, and churn rates are descriptive
associations and do not prove causation.
