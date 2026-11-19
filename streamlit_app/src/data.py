from io import BytesIO
from pathlib import Path

import pandas as pd

TARGET = "Churn Value"
NUMERIC_FEATURES = ["Tenure Months", "Monthly Charges", "Total Charges"]
CATEGORICAL_FEATURES = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service",
    "Multiple Lines", "Internet Service", "Online Security", "Online Backup",
    "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies",
    "Contract", "Paperless Billing", "Payment Method",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]
LEAKAGE_COLUMNS = ["Churn Label", "Churn Reason", "Churn Score", "CLTV"]
WORKBOOK_NAME = "Telco_customer_churn.xlsx"

# Fail loudly if a leakage field is accidentally added to the predictors later.
_leaked_features = set(LEAKAGE_COLUMNS + [TARGET]) & set(FEATURES)
if _leaked_features:
    raise RuntimeError(
        f"Target/leakage columns used as predictors: {sorted(_leaked_features)}"
    )


class DatasetError(ValueError):
    """Raised when the uploaded workbook is missing expected data."""


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DatasetError(
            "The workbook is missing required columns: " + ", ".join(missing)
        )
    data = frame.copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # The source has blank Total Charges for customers with zero tenure.
    zero_tenure_blanks = data["Total Charges"].isna() & data["Tenure Months"].eq(0)
    data.loc[zero_tenure_blanks, "Total Charges"] = 0.0
    if data[TARGET].isna().any() or not data[TARGET].isin([0, 1]).all():
        raise DatasetError("Churn Value must contain only 0/1 values with no blanks.")
    return data


def load_workbook(source) -> pd.DataFrame:
    if isinstance(source, (str, Path)) and not Path(source).exists():
        raise FileNotFoundError(f"Dataset workbook not found: {source}")
    try:
        if isinstance(source, (bytes, bytearray)):
            source = BytesIO(source)
        return clean_data(pd.read_excel(source))
    except DatasetError:
        raise
    except Exception as exc:
        if isinstance(exc, ImportError) or "openpyxl" in str(exc).lower():
            raise DatasetError(
                "Reading .xlsx files requires openpyxl. Install the project dependencies "
                "in the same Python environment used to start Streamlit."
            ) from exc
        raise DatasetError(
            f"Could not read this file as an Excel workbook ({exc}). "
            "Check that it is a valid .xlsx file with the expected columns."
        ) from exc


def default_workbook_path() -> Path:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.parents[2] / WORKBOOK_NAME,
        Path.cwd() / WORKBOOK_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Keep a useful expected path for the app's missing-file message.
    return candidates[0]
