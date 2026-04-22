
# %%
import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(page_title="Supply Chain Risk AI", page_icon="📦", layout="wide")
st.title("📦 Procurement Risk Modeling App")
st.caption("Upload procurement and coordinates files, engineer features, and train early-warning models.")
st.markdown(
    """
    <style>
    :root {
        --bg-main: #f3f7fb;
        --bg-card: rgba(255, 255, 255, 0.9);
        --bg-card-strong: #ffffff;
        --border-soft: #d6e2f0;
        --text-main: #10233f;
        --text-muted: #5c6f87;
        --accent: #1f5eff;
        --shadow-soft: 0 16px 40px rgba(15, 23, 42, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(31, 94, 255, 0.12), transparent 24%),
            radial-gradient(circle at top right, rgba(15, 159, 110, 0.08), transparent 20%),
            linear-gradient(180deg, #f8fbff 0%, var(--bg-main) 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #112544 0%, #16355d 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f8fbff;
    }

    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px dashed rgba(255, 255, 255, 0.28);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%);
        color: #f8fbff;
        border: 1px solid rgba(255, 255, 255, 0.16);
        font-weight: 700;
        box-shadow: 0 10px 22px rgba(8, 47, 73, 0.28);
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        background: linear-gradient(135deg, #22c55e 0%, #0f766e 100%);
        border-color: rgba(255, 255, 255, 0.28);
        color: #ffffff;
    }

    [data-testid="stSidebar"] [data-testid="stButton"] button:disabled {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.45) 0%, rgba(15, 118, 110, 0.45) 100%);
        color: rgba(248, 251, 255, 0.82);
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: none;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0);
    }

    h1 {
        color: var(--text-main);
        font-size: 2.55rem !important;
        line-height: 1.05;
        letter-spacing: -0.04em;
        margin-bottom: 0.3rem;
    }

    .stCaptionContainer {
        background: linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(238,244,251,0.92) 100%);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow-soft);
        margin-bottom: 1rem;
    }

    .section-intro {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        margin-bottom: 1rem;
    }

    .section-intro h3 {
        margin: 0 0 0.25rem 0;
        color: var(--text-main);
        font-size: 1.15rem;
    }

    .section-intro p {
        margin: 0;
        color: var(--text-muted);
        font-size: 0.95rem;
    }

    [data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin-bottom: 0.8rem;
    }

    [data-baseweb="tab"] {
        height: 44px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid var(--border-soft);
        padding: 0 1rem;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        background: linear-gradient(135deg, var(--accent) 0%, #3b82f6 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 10px 22px rgba(31, 94, 255, 0.24);
    }

    [data-testid="stMetric"] {
        background: var(--bg-card);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: var(--shadow-soft);
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-muted);
    }

    [data-testid="stMetricValue"] {
        color: var(--text-main);
    }

    [data-testid="stDataFrame"], div[data-testid="stTable"] {
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        overflow: hidden;
        background: var(--bg-card-strong);
        box-shadow: var(--shadow-soft);
    }

    [data-testid="stForm"] {
        background: var(--bg-card);
        border: 1px solid var(--border-soft);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 0.8rem 1.2rem;
        box-shadow: var(--shadow-soft);
    }

    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stButton"] button {
        border-radius: 12px;
        border: 1px solid #c9dafc;
        background: linear-gradient(135deg, #1f5eff 0%, #2563eb 100%);
        color: #ffffff;
        font-weight: 700;
        box-shadow: 0 8px 20px rgba(31, 94, 255, 0.18);
    }

    .result-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(245,248,252,0.98) 100%);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 134px;
        box-shadow: var(--shadow-soft);
    }

    .result-card-label {
        color: var(--text-muted);
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .result-card-value {
        margin-top: 0.55rem;
        color: var(--text-main);
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }

    .result-card-note {
        margin-top: 0.55rem;
        color: var(--text-muted);
        font-size: 0.88rem;
        line-height: 1.45;
    }

    .callout-banner {
        background: linear-gradient(90deg, #e8f0ff 0%, #eef9f4 100%);
        border: 1px solid var(--border-soft);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        color: var(--text-main);
        margin: 0.9rem 0 1rem 0;
        box-shadow: var(--shadow-soft);
    }

    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.38rem 0.72rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        margin-right: 0.45rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-intro" style="margin-top:0.35rem; margin-bottom:1.2rem;">
        <h3>Executive Procurement Control Tower</h3>
        <p>
            Use this workspace to engineer procurement data, monitor supplier performance, train delay-risk models,
            and present route-level decisions in a more polished and stakeholder-ready format.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    r = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def classify_distance(distance):
    if pd.isna(distance):
        return np.nan
    if distance < 100:
        return "short"
    if distance <= 500:
        return "medium"
    return "long"


def weather_risk_from_distance(distance_km):
    """
    Placeholder risk if weather API is not included.
    Replace later with your real weather-enriched logic if needed.
    """
    if pd.isna(distance_km):
        return "unknown"
    if distance_km > 1000:
        return "high"
    if distance_km > 300:
        return "medium"
    return "low"


def delay_category(days):
    if pd.isna(days):
        return np.nan
    if days <= 0:
        return "On-time"
    if 1 <= days <= 7:
        return "Minor delay (1-7 days)"
    if 8 <= days <= 30:
        return "Medium delay (8-30 days)"
    return "Major delay (>30 days)"


def build_preprocessor(x):
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])


def safe_read_excel(source, **kwargs):
    if isinstance(source, (str, Path)):
        return pd.read_excel(source, **kwargs)

    source.seek(0)
    file_bytes = source.read()
    return pd.read_excel(io.BytesIO(file_bytes), **kwargs)


def validate_sheet_columns(df, required_cols, label):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {label}: {missing}")


def get_default_data_sources():
    procurement_candidates = [
        DATA_DIR / "Datasets_Procurement_Cleaned_20260211_165716.xlsx",
        DATA_DIR / "Datasets_Procurement_Cleaned_20260210_210209.xlsx",
        DATA_DIR / "Datasets Procurement_Cleaned.xlsx",
        DATA_DIR / "Datasets Procurement.xlsx",
    ]
    coords_candidates = [DATA_DIR / "Private_coordinates.xlsx"]

    procurement_path = next((path for path in procurement_candidates if path.exists()), None)
    coords_path = next((path for path in coords_candidates if path.exists()), None)
    return procurement_path, coords_path

def classify_transport_region(distance_km):
    if pd.isna(distance_km):
        return "unknown"
    if distance_km <= 500:
        return "domestic"
    return "interstate"


def classify_transport_corridor_risk(distance_km):
    if pd.isna(distance_km):
        return "unknown"
    if distance_km > 1000:
        return "high"
    if distance_km > 300:
        return "medium"
    return "low"


def capped_ratio(numerator, denominator):
    if pd.isna(denominator) or denominator <= 0:
        return 0.0
    if pd.isna(numerator):
        return 0.0
    return float(np.clip(numerator / denominator, 0, 1))


def lateness_bucket_score(days_late):
    if pd.isna(days_late) or days_late <= 5:
        return 0.0
    if days_late <= 20:
        return 0.5
    if days_late <= 60:
        return 0.75
    return 1.0


def add_supplier_delivery_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    today = pd.Timestamp.today().normalize()

    reference_receipt_date = df["ReceiptDate"].fillna(today)
    risk_days_late = (reference_receipt_date - df["RequiredDate"]).dt.days
    risk_days_late = risk_days_late.fillna(0)

    df["supplier_risk_days_late"] = risk_days_late
    df["late_item_flag"] = (df["supplier_risk_days_late"] > 0).astype(int)
    df["late_bucket_score"] = df["supplier_risk_days_late"].apply(lateness_bucket_score)

    late_rows = df["late_item_flag"] == 1
    overall_extension_amount = pd.to_numeric(df["ExtensionAmt"], errors="coerce").fillna(0).sum()

    supplier_metrics = (
        df.groupby("PurchaseName", dropna=False)
        .agg(
            supplier_total_items=("ItemCode", "size"),
            supplier_late_items=("late_item_flag", "sum"),
            total_extension_amount=("ExtensionAmt", "sum"),
        )
        .reset_index()
    )

    severity_scores = (
        df.loc[late_rows]
        .groupby("PurchaseName", dropna=False)["late_bucket_score"]
        .mean()
        .rename("lateness_severity_score")
        .reset_index()
    )

    supplier_metrics = supplier_metrics.merge(
        severity_scores,
        on="PurchaseName",
        how="left",
    )
    supplier_metrics["lateness_severity_score"] = (
        supplier_metrics["lateness_severity_score"].fillna(0).clip(0, 1)
    )
    supplier_metrics["late_rate"] = supplier_metrics.apply(
        lambda row: capped_ratio(row["supplier_late_items"], row["supplier_total_items"]),
        axis=1,
    )
    supplier_metrics["extension_rate"] = supplier_metrics["total_extension_amount"].apply(
        lambda value: capped_ratio(value, overall_extension_amount)
    )
    supplier_metrics["delivery_risk_score"] = (
        supplier_metrics["late_rate"] * 0.40
        + supplier_metrics["extension_rate"] * 0.20
        + supplier_metrics["lateness_severity_score"] * 0.40
    ).clip(0, 1)

    round_cols = [
        "late_rate",
        "extension_rate",
        "lateness_severity_score",
        "delivery_risk_score",
    ]
    supplier_metrics[round_cols] = supplier_metrics[round_cols].round(4)

    df = df.merge(
        supplier_metrics[
            [
                "PurchaseName",
                "late_rate",
                "extension_rate",
                "lateness_severity_score",
                "delivery_risk_score",
            ]
        ],
        on="PurchaseName",
        how="left",
    )

    return df.drop(
        columns=["supplier_risk_days_late", "late_item_flag", "late_bucket_score"],
        errors="ignore",
    )


def render_section_intro(title, description):
    st.markdown(
        f"""
        <div class="section-intro">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_card(label, value, note):
    note_html = f'<div class="result-card-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-card-label">{label}</div>
            <div class="result-card-value">{value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def mode_or_unknown(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "unknown"
    mode = values.mode()
    if mode.empty:
        return "unknown"
    return str(mode.iloc[0])


def can_stratify_target(y):
    class_counts = y.value_counts(dropna=False)
    return y.nunique() > 1 and not class_counts.empty and class_counts.min() >= 2


def get_positive_class_probability(model, x):
    probabilities = model.predict_proba(x)
    class_labels = getattr(model, "classes_", [])

    if 1 in class_labels:
        positive_class_index = list(class_labels).index(1)
        return probabilities[:, positive_class_index]

    return np.zeros(len(x), dtype=float)
# ============================================================
# DATA ENGINEERING
# ============================================================
@st.cache_data(show_spinner=False, ttl=0)
def process_procurement(procurement_file, private_coords_file):
    df_proc = safe_read_excel(procurement_file)
    df_proc.columns = df_proc.columns.str.strip()

    required_proc_cols = [
        "ShipToCode",
        "PurchaseName",
        "ReceiptDate",
        "RequiredDate",
        "QuantityOrdered",
        "QuantityReceived",
        "PurchaseOrderDate",
        "UnitCost",
        "ItemCode",
        "ExtensionAmt",
    ]
    validate_sheet_columns(df_proc, required_proc_cols, "procurement file")

    df_proc = df_proc[df_proc["ShipToCode"].notna() & df_proc["PurchaseName"].notna()].copy()
    df_proc["ShipToCode"] = df_proc["ShipToCode"].astype(str).str.strip()
    df_proc["PurchaseName"] = df_proc["PurchaseName"].astype(str).str.strip()
    df_proc["ItemCode"] = df_proc["ItemCode"].astype(str).str.strip()

    numeric_cols = ["QuantityOrdered", "QuantityReceived", "UnitCost", "ExtensionAmt"]
    for col in numeric_cols:
        df_proc[col] = pd.to_numeric(df_proc[col], errors="coerce")

    # The private coordinates workbook currently keeps swapped tab labels:
    # - "Destinations" contains origin location records keyed by ShipToCode
    # - "Origins" contains supplier destination records keyed by PurchaseName
    df_origins = safe_read_excel(private_coords_file, sheet_name="Destinations")
    df_origins.columns = df_origins.columns.str.strip().str.lower()
    validate_sheet_columns(df_origins, ["location", "lat_private", "lon_private"], "Destinations sheet")

    lookup_origins = df_origins.rename(columns={
        "lat_private": "origin_latitude",
        "lon_private": "origin_longitude",
    }).copy()

    lookup_origins["location"] = lookup_origins["location"].astype(str).str.strip()
    lookup_origins["origin_latitude"] = pd.to_numeric(lookup_origins["origin_latitude"], errors="coerce")
    lookup_origins["origin_longitude"] = pd.to_numeric(lookup_origins["origin_longitude"], errors="coerce")

    df_proc = df_proc.merge(
        lookup_origins[["location", "origin_latitude", "origin_longitude"]],
        how="left",
        left_on="ShipToCode",
        right_on="location",
    )
    df_proc.drop(columns=["location"], inplace=True, errors="ignore")

    # Destinations
    df_dest = safe_read_excel(private_coords_file, sheet_name="Origins")
    df_dest.columns = df_dest.columns.str.strip().str.lower()
    validate_sheet_columns(df_dest, ["location", "lat_private", "lon_private"], "Origins sheet")

    lookup_dest = df_dest.rename(columns={
        "lat_private": "latitude_destination",
        "lon_private": "longitude_destination",
    }).copy()

    lookup_dest["location"] = lookup_dest["location"].astype(str).str.strip()
    lookup_dest["latitude_destination"] = pd.to_numeric(lookup_dest["latitude_destination"], errors="coerce")
    lookup_dest["longitude_destination"] = pd.to_numeric(lookup_dest["longitude_destination"], errors="coerce")

    df_final = df_proc.merge(
        lookup_dest[["location", "latitude_destination", "longitude_destination"]],
        how="left",
        left_on="PurchaseName",
        right_on="location",
    )
    df_final.drop(columns=["location"], inplace=True, errors="ignore")

    df_final = df_final.dropna(subset=[
        "origin_latitude",
        "origin_longitude",
        "latitude_destination",
        "longitude_destination"
    ]).copy()

    # Distance
    df_final["distance_km"] = haversine(
        df_final["origin_latitude"],
        df_final["origin_longitude"],
        df_final["latitude_destination"],
        df_final["longitude_destination"],
    )
    df_final["distance_category"] = df_final["distance_km"].apply(classify_distance)

    # Dates
    for col in ["ReceiptDate", "RequiredDate", "PurchaseOrderDate"]:
        df_final[col] = pd.to_datetime(df_final[col], errors="coerce")

    df_final = df_final.dropna(subset=["RequiredDate", "PurchaseOrderDate"]).copy()

    # Delay features
    df_final["late_days"] = (df_final["ReceiptDate"] - df_final["RequiredDate"]).dt.days
    df_final["is_late"] = (df_final["late_days"] > 0).astype(int)
    df_final["delay_category"] = df_final["late_days"].apply(delay_category)

    # Fulfillment features
    df_final["Backordered_qty"] = df_final["QuantityOrdered"] - df_final["QuantityReceived"]
    df_final["DeliveryStatus"] = np.where(
        df_final["Backordered_qty"] == 0,
        "Fully delivered",
        "Incomplete order",
    )
    df_final["fulfillment_rate"] = np.where(
        df_final["QuantityOrdered"] > 0,
        df_final["QuantityReceived"] / df_final["QuantityOrdered"],
        np.nan,
    )

    # Core engineered features
    df_final["order_value"] = df_final["UnitCost"] * df_final["QuantityOrdered"]

    count_col = "PO Key" if "PO Key" in df_final.columns else "ItemCode"
    df_final["item_order_frequency"] = df_final.groupby("ItemCode")[count_col].transform("count")
    df_final["supplier_item_frequency"] = (
        df_final.groupby(["PurchaseName", "ItemCode"])[count_col].transform("count")
    )

    df_final["order_month"] = df_final["PurchaseOrderDate"].dt.month
    df_final["order_day_of_week"] = df_final["PurchaseOrderDate"].dt.dayofweek + 1
    df_final["lead_time_days"] = (
        df_final["RequiredDate"] - df_final["PurchaseOrderDate"]
    ).dt.days

    # Price features
    df_final["historical_item_avg_price"] = df_final.groupby("ItemCode")["UnitCost"].transform("mean")
    df_final["price_variance"] = df_final["UnitCost"] - df_final["historical_item_avg_price"]
    df_final["price_ratio"] = np.where(
        df_final["historical_item_avg_price"] != 0,
        df_final["UnitCost"] / df_final["historical_item_avg_price"],
        np.nan,
    )
    df_final["supplier_price_rank"] = (
        df_final.groupby("ItemCode")["UnitCost"].rank(method="dense", ascending=True)
    )

    df_final = add_supplier_delivery_risk_features(df_final)

    # Lightweight placeholders
# Engineered transport features
    df_final["weather_risk_index"] = df_final["distance_km"].apply(weather_risk_from_distance)
    df_final["transport_region"] = df_final["distance_km"].apply(classify_transport_region)
    df_final["transport_corridor_risk"] = df_final["distance_km"].apply(classify_transport_corridor_risk)

    return df_final


# ============================================================
# MODELING
# ============================================================
@st.cache_resource(show_spinner=False)
def train_models(df):
    base_features = [
        "distance_km",
        "lead_time_days",
        "order_value",
        "item_order_frequency",
        "supplier_item_frequency",
        "order_month",
        "order_day_of_week",
        "price_ratio",
        "delivery_risk_score",
    ]

    optional_numeric = ["precipitation", "max_wind"]
    base_features.extend([c for c in optional_numeric if c in df.columns])

    categorical_features = [
        c for c in ["weather_risk_index", "transport_region", "transport_corridor_risk"]
        if c in df.columns
    ]

    feature_list = [c for c in base_features + categorical_features if c in df.columns]

    if not feature_list:
        raise ValueError("No valid modeling features found.")

    modeling_df = df.dropna(subset=["late_days", "is_late"]).copy()
    if modeling_df.empty:
        raise ValueError("No rows are available for modeling after removing missing targets.")

    x = modeling_df[feature_list]
    y_reg = modeling_df["late_days"]
    y_clf = modeling_df["is_late"]

    if len(modeling_df) < 10:
        raise ValueError("At least 10 modeled rows are needed to train the baseline models.")
    if y_clf.nunique() < 2:
        raise ValueError("The target contains only one class. Include both on-time and late deliveries.")

    stratify_target = y_clf if can_stratify_target(y_clf) else None

    x_train, x_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        x, y_reg, y_clf, test_size=0.2, random_state=42, stratify=stratify_target
    )

    if y_train_clf.nunique() < 2:
        raise ValueError(
            "The training split ended up with one class. Add more history or use a more balanced dataset."
        )

    preprocessor = build_preprocessor(x_train)

    reg_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
        ))
    ])

    clf_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
        ))
    ])

    reg_pipe.fit(x_train, y_train_reg)
    clf_pipe.fit(x_train, y_train_clf)

    reg_pred = reg_pipe.predict(x_test)
    clf_pred = clf_pipe.predict(x_test)
    clf_prob = get_positive_class_probability(clf_pipe, x_test)

    reg_metrics = {
        "MAE": float(mean_absolute_error(y_test_reg, reg_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_reg, reg_pred))),
        "R2": float(r2_score(y_test_reg, reg_pred)),
    }

    clf_metrics = {
        "Accuracy": float(accuracy_score(y_test_clf, clf_pred)),
        "ROC_AUC": float(roc_auc_score(y_test_clf, clf_prob)) if y_test_clf.nunique() > 1 else np.nan,
        "Confusion_Matrix": confusion_matrix(y_test_clf, clf_pred),
        "Classification_Report": classification_report(y_test_clf, clf_pred, output_dict=True, zero_division=0),
    }

    feature_importance_df = None
    if hasattr(clf_pipe.named_steps["model"], "feature_importances_"):
        feature_names = clf_pipe.named_steps["preprocessor"].get_feature_names_out()
        importances = clf_pipe.named_steps["model"].feature_importances_
        feature_importance_df = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    predictions_df = x_test.copy()
    predictions_df["actual_late_days"] = y_test_reg.values
    predictions_df["predicted_late_days"] = reg_pred
    predictions_df["actual_is_late"] = y_test_clf.values
    predictions_df["predicted_is_late"] = clf_pred
    predictions_df["delay_risk_probability"] = clf_prob

    return {
        "feature_list": feature_list,
        "reg_metrics": reg_metrics,
        "clf_metrics": clf_metrics,
        "feature_importance_df": feature_importance_df,
        "predictions_df": predictions_df,
        "reg_model": reg_pipe,
        "clf_model": clf_pipe,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_balance": y_clf.value_counts().sort_index().to_dict(),
        "used_stratified_split": bool(stratify_target is not None),
    }


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### Data Intake")
    st.caption("Load the source files used to build the procurement intelligence workspace.")
    procurement_file = st.file_uploader("Upload procurement Excel file", type=["xlsx"])
    private_coords_file = st.file_uploader("Upload private coordinates Excel file", type=["xlsx"])
    run_processing = st.button("Run data engineering", type="primary")
    run_sample_processing = st.button("Use repo sample files")

    default_procurement_path, default_coords_path = get_default_data_sources()
    if default_procurement_path and default_coords_path:
        st.caption(
            f"Bundled files detected: `{default_procurement_path.name}` and `{default_coords_path.name}`."
        )
    else:
        st.caption("Bundled sample files were not fully detected in the local `data` folder.")


# ============================================================
# SESSION STATE
# ============================================================
if "engineered_df" not in st.session_state:
    st.session_state.engineered_df = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "predicted_route_df" not in st.session_state:
    st.session_state.predicted_route_df = None


# ============================================================
# MAIN APP
# ============================================================
if run_processing:
    if procurement_file is None or private_coords_file is None:
        st.error("Please upload both Excel files.")
    else:
        try:
            with st.spinner("Processing uploaded files..."):
                engineered_df = process_procurement(procurement_file, private_coords_file)
                st.session_state.engineered_df = engineered_df
                st.session_state.model_results = None
                st.session_state.predicted_route_df = None
            st.success("Data engineering completed.")
        except Exception as e:
            st.exception(e)

if run_sample_processing:
    default_procurement_path, default_coords_path = get_default_data_sources()
    if default_procurement_path is None or default_coords_path is None:
        st.error("Bundled sample files were not found in the local data folder.")
    else:
        try:
            with st.spinner("Processing bundled sample files..."):
                engineered_df = process_procurement(default_procurement_path, default_coords_path)
                st.session_state.engineered_df = engineered_df
                st.session_state.model_results = None
                st.session_state.predicted_route_df = None
            st.success("Bundled sample files processed successfully.")
        except Exception as e:
            st.exception(e)


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Engineered Data", "KPIs", "Modeling", "Single Prediction", "Map"]
)

with tab1:
    render_section_intro(
        "Engineered Dataset",
        "Review the modeled procurement dataset, confirm record counts, and export the enriched output for downstream analysis.",
    )
    if st.session_state.engineered_df is None:
        st.info("Upload files and click 'Run data engineering' to generate the dataset.")
    else:
        df = st.session_state.engineered_df
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]:,}")
        c3.metric("Late deliveries", f"{int(df['is_late'].sum()):,}")

        st.dataframe(df.head(100), use_container_width=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download engineered CSV",
            data=csv_data,
            file_name="CleanandEngineered_Data.csv",
            mime="text/csv",
        )

with tab2:
    render_section_intro(
        "Operational KPIs",
        "Track delivery performance, on-time fulfillment, and supplier reliability through a cleaner operations dashboard.",
    )
    if st.session_state.engineered_df is None:
        st.info("Generate the engineered dataset first.")
    else:
        df = st.session_state.engineered_df.copy()
        supplier_options = ["All suppliers"]
        if "PurchaseName" in df.columns:
            supplier_options += sorted(df["PurchaseName"].dropna().astype(str).unique().tolist())
        selected_supplier = st.selectbox(
            "Supplier filter",
            supplier_options,
            key="kpi_supplier_filter",
        )
        kpi_df = (
            df.copy()
            if selected_supplier == "All suppliers"
            else df[df["PurchaseName"].astype(str) == selected_supplier].copy()
        )

        total_deliveries = len(kpi_df)
        on_time_deliveries = int((kpi_df["is_late"] == 0).sum())
        late_deliveries = int((kpi_df["is_late"] == 1).sum())
        otd = (on_time_deliveries / total_deliveries * 100) if total_deliveries else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total deliveries", f"{total_deliveries:,}")
        c2.metric("On-time deliveries", f"{on_time_deliveries:,}")
        c3.metric("Late deliveries", f"{late_deliveries:,}")
        c4.metric("OTD %", f"{otd:.2f}%")

        risk_cols = [
            "late_rate",
            "extension_rate",
            "lateness_severity_score",
            "delivery_risk_score",
        ]
        if all(col in kpi_df.columns for col in risk_cols):
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Avg late_rate", f"{kpi_df['late_rate'].mean():.4f}")
            r2.metric("Avg extension_rate", f"{kpi_df['extension_rate'].mean():.4f}")
            r3.metric("Avg lateness_severity", f"{kpi_df['lateness_severity_score'].mean():.4f}")
            r4.metric("Avg delivery_risk", f"{kpi_df['delivery_risk_score'].mean():.4f}")

        left, right = st.columns(2)

        with left:
            st.markdown("**Delivery status summary**")
            status_summary = kpi_df["DeliveryStatus"].value_counts(dropna=False).reset_index()
            status_summary.columns = ["DeliveryStatus", "Count"]
            st.dataframe(status_summary, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            kpi_df["DeliveryStatus"].value_counts().plot(kind="bar", ax=ax, color="#0f766e")
            ax.set_title("Delivery Status")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        with right:
            st.markdown("**Late vs On-time**")
            late_summary = kpi_df["is_late"].map({1: "Late", 0: "On Time"}).value_counts().reset_index()
            late_summary.columns = ["Status", "Count"]
            st.dataframe(late_summary, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 4))
            kpi_df["is_late"].map({1: "Late", 0: "On Time"}).value_counts().plot(kind="bar", ax=ax, color="#1f5eff")
            ax.set_title("Late vs On Time")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        if "PurchaseName" in df.columns:
            st.markdown("**Supplier OTD**")
            supplier_otd = df.groupby("PurchaseName").agg(
                total_deliveries=("PurchaseName", "count"),
                late_deliveries=("is_late", "sum"),
            )
            supplier_otd["on_time_deliveries"] = (
                supplier_otd["total_deliveries"] - supplier_otd["late_deliveries"]
            )
            supplier_otd["OTD_%"] = (
                supplier_otd["on_time_deliveries"] / supplier_otd["total_deliveries"] * 100
            )
            if all(col in df.columns for col in risk_cols):
                supplier_risk = df.groupby("PurchaseName").agg(
                    late_rate=("late_rate", "max"),
                    extension_rate=("extension_rate", "max"),
                    lateness_severity_score=("lateness_severity_score", "max"),
                    delivery_risk_score=("delivery_risk_score", "max"),
                )
                supplier_otd = supplier_otd.merge(
                    supplier_risk,
                    left_index=True,
                    right_index=True,
                    how="left",
                )

            route_cols = [
                "weather_risk_index",
                "transport_region",
                "transport_corridor_risk",
            ]
            available_route_cols = [col for col in route_cols if col in df.columns]
            if available_route_cols:
                supplier_route = df.groupby("PurchaseName").agg(
                    {
                        col: mode_or_unknown
                        for col in available_route_cols
                    }
                )
                supplier_otd = supplier_otd.merge(
                    supplier_route,
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            supplier_otd = supplier_otd.reset_index().sort_values("OTD_%", ascending=False)
            supplier_view = (
                supplier_otd
                if selected_supplier == "All suppliers"
                else supplier_otd[supplier_otd["PurchaseName"] == selected_supplier]
            )

            st.markdown("**Supplier KPI report**")
            heatmap_cols = [
                col for col in [
                    "OTD_%",
                    "late_rate",
                    "extension_rate",
                    "lateness_severity_score",
                    "delivery_risk_score",
                ]
                if col in supplier_view.columns
            ]
            supplier_view_display = supplier_view.copy()
            if heatmap_cols:
                supplier_view_display = supplier_view_display.style.background_gradient(
                    cmap="RdYlGn_r",
                    subset=heatmap_cols,
                ).format(
                    {
                        "OTD_%": "{:.2f}",
                        "late_rate": "{:.4f}",
                        "extension_rate": "{:.4f}",
                        "lateness_severity_score": "{:.4f}",
                        "delivery_risk_score": "{:.4f}",
                    }
                )
            st.dataframe(supplier_view_display, use_container_width=True)

            numeric_compare_cols = [
                "late_rate",
                "extension_rate",
                "lateness_severity_score",
                "delivery_risk_score",
            ]
            available_compare_cols = [col for col in numeric_compare_cols if col in supplier_otd.columns]
            if available_compare_cols:
                st.markdown("**Supplier comparison charts**")
                compare_col1, compare_col2 = st.columns(2)

                top_compare = supplier_otd.nlargest(10, "delivery_risk_score")[
                    ["PurchaseName", "delivery_risk_score"]
                ]
                with compare_col1:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    color_norm = plt.Normalize(vmin=0, vmax=max(0.8, float(top_compare["delivery_risk_score"].max())))
                    heatmap_colors = plt.cm.YlOrRd(
                        color_norm(top_compare["delivery_risk_score"].to_numpy())
                    )
                    ax.barh(
                        top_compare["PurchaseName"],
                        top_compare["delivery_risk_score"],
                        color=heatmap_colors,
                    )
                    ax.set_title("Top 10 suppliers by delivery risk score")
                    ax.set_xlabel("delivery_risk_score")
                    ax.invert_yaxis()
                    st.pyplot(fig)

                compare_df = supplier_otd.set_index("PurchaseName")[available_compare_cols]
                if selected_supplier != "All suppliers" and selected_supplier in compare_df.index:
                    compare_chart_df = pd.concat(
                        [
                            compare_df.mean().rename("Portfolio avg"),
                            compare_df.loc[selected_supplier].rename(selected_supplier),
                        ],
                        axis=1,
                    ).T
                    chart_title = f"{selected_supplier} vs portfolio average"
                else:
                    compare_chart_df = compare_df.sort_values(
                        "delivery_risk_score", ascending=False
                    ).head(8)
                    chart_title = "Supplier risk comparison"

                with compare_col2:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    compare_chart_df.plot(kind="bar", ax=ax)
                    ax.set_title(chart_title)
                    ax.set_ylabel("Score")
                    ax.legend(loc="upper right", fontsize=8)
                    plt.xticks(rotation=25, ha="right")
                    plt.tight_layout()
                    st.pyplot(fig)

with tab3:
    render_section_intro(
        "Early-Warning Modeling",
        "Train the classification and regression models, review their performance, and inspect the strongest risk drivers.",
    )
    if st.session_state.engineered_df is None:
        st.info("Generate the engineered dataset first.")
    else:
        df = st.session_state.engineered_df.copy()
        if st.button("Train models"):
            try:
                with st.spinner("Training models..."):
                    results = train_models(df)
                    st.session_state.model_results = results

                st.success("Models trained successfully.")

                st.markdown("**Features used**")
                st.write(results["feature_list"])

                balance_labels = {0: "On-time", 1: "Late"}
                class_balance_text = ", ".join(
                    f"{balance_labels.get(label, label)}: {count:,}"
                    for label, count in results["class_balance"].items()
                )
                st.caption(
                    f"Train rows: {results['train_rows']:,} | Test rows: {results['test_rows']:,} | "
                    f"Class balance: {class_balance_text} | "
                    f"Stratified split: {'Yes' if results['used_stratified_split'] else 'No'}"
                )

                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("MAE", f"{results['reg_metrics']['MAE']:.3f}")
                col2.metric("RMSE", f"{results['reg_metrics']['RMSE']:.3f}")
                col3.metric("R²", f"{results['reg_metrics']['R2']:.3f}")
                col4.metric("Accuracy", f"{results['clf_metrics']['Accuracy']:.3f}")
                roc_auc_value = results["clf_metrics"]["ROC_AUC"]
                col5.metric("ROC-AUC", "N/A" if pd.isna(roc_auc_value) else f"{roc_auc_value:.3f}")

                if pd.isna(roc_auc_value):
                    st.info("ROC-AUC is unavailable because the test split contained only one target class.")

                st.markdown("**Confusion matrix**")
                cm = pd.DataFrame(
                    results["clf_metrics"]["Confusion_Matrix"],
                    index=["Actual 0", "Actual 1"],
                    columns=["Pred 0", "Pred 1"],
                )
                st.dataframe(cm, use_container_width=True)

                predictions_df = results["predictions_df"].copy()
                chart_left, chart_right = st.columns(2)

                with chart_left:
                    st.markdown("**ROC curve**")
                    if predictions_df["actual_is_late"].nunique() > 1:
                        fpr, tpr, _ = roc_curve(
                            predictions_df["actual_is_late"],
                            predictions_df["delay_risk_probability"],
                        )
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(fpr, tpr, color="#dc2626", linewidth=2, label="Model ROC")
                        ax.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Baseline")
                        ax.set_title("ROC curve")
                        ax.set_xlabel("False Positive Rate")
                        ax.set_ylabel("True Positive Rate")
                        ax.legend()
                        st.pyplot(fig)
                        st.caption(
                            "Interpretation: Curves closer to the top-left corner indicate stronger separation "
                            "between late and on-time deliveries. Higher ROC-AUC means better ranking of risky orders."
                        )
                    else:
                        st.info("ROC curve is unavailable because the test sample contains one class.")

                with chart_right:
                    st.markdown("**Precision-recall curve**")
                    if predictions_df["actual_is_late"].nunique() > 1:
                        precision, recall, _ = precision_recall_curve(
                            predictions_df["actual_is_late"],
                            predictions_df["delay_risk_probability"],
                        )
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.plot(recall, precision, color="#f59e0b", linewidth=2)
                        ax.set_title("Precision-recall curve")
                        ax.set_xlabel("Recall")
                        ax.set_ylabel("Precision")
                        st.pyplot(fig)
                        st.caption(
                            "Interpretation: This shows how well the model balances catching late deliveries "
                            "with avoiding false alarms. Higher precision at strong recall is better."
                        )
                    else:
                        st.info("Precision-recall curve is unavailable because the test sample contains one class.")

                reg_chart_left, reg_chart_right = st.columns(2)
                residuals = (
                    predictions_df["actual_late_days"] - predictions_df["predicted_late_days"]
                )

                with reg_chart_left:
                    st.markdown("**Actual vs predicted late days**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(
                        predictions_df["actual_late_days"],
                        predictions_df["predicted_late_days"],
                        alpha=0.55,
                        color="#1f5eff",
                    )
                    line_min = min(
                        predictions_df["actual_late_days"].min(),
                        predictions_df["predicted_late_days"].min(),
                    )
                    line_max = max(
                        predictions_df["actual_late_days"].max(),
                        predictions_df["predicted_late_days"].max(),
                    )
                    ax.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#ef4444")
                    ax.set_title("Regression fit")
                    ax.set_xlabel("Actual late days")
                    ax.set_ylabel("Predicted late days")
                    st.pyplot(fig)
                    st.caption(
                        "Interpretation: Points closer to the diagonal line indicate more accurate delay-length "
                        "predictions. Wide scatter means the model has more error estimating exact late days."
                    )

                with reg_chart_right:
                    st.markdown("**Residual distribution**")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.hist(residuals, bins=30, color="#0f766e", edgecolor="white")
                    ax.axvline(0, linestyle="--", color="#ef4444")
                    ax.set_title("Residuals")
                    ax.set_xlabel("Actual - predicted late days")
                    ax.set_ylabel("Count")
                    st.pyplot(fig)
                    st.caption(
                        "Interpretation: Residuals centered near zero suggest the regression model is broadly "
                        "well calibrated. Strong skew or very wide spread suggests systematic over- or underprediction."
                    )

                if results["feature_importance_df"] is not None:
                    st.markdown("**Top 15 feature importances**")
                    top_features = results["feature_importance_df"].head(15).sort_values("importance")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.barh(top_features["feature"], top_features["importance"])
                    ax.set_title("Top 15 Features Driving Delay Risk")
                    ax.set_xlabel("Importance")
                    st.pyplot(fig)
                    st.caption(
                        "Interpretation: Taller importance bars indicate features that contribute more strongly "
                        "to the model's delay-risk decisions."
                    )
                    st.dataframe(results["feature_importance_df"].head(25), use_container_width=True)

                st.markdown("**Prediction sample**")
                st.dataframe(predictions_df.head(100), use_container_width=True)

                pred_csv = predictions_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download prediction sample CSV",
                    data=pred_csv,
                    file_name="model_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.exception(e)

with tab4:
    render_section_intro(
        "Single Order Delay Prediction",
        "Score one order at a time using the trained models and surface a decision-ready risk readout for planners.",
    )
    if st.session_state.engineered_df is None:
        st.info("Generate the engineered dataset first.")
    else:
        df = st.session_state.engineered_df.copy()

        if st.session_state.model_results is None:
            st.warning("Train models in the Modeling tab before using this prediction form.")
        else:
            suppliers = sorted(df["PurchaseName"].dropna().astype(str).unique().tolist()) if "PurchaseName" in df.columns else []
            purchase_series = df["PurchaseName"].astype(str) if "PurchaseName" in df.columns else pd.Series(dtype=str)
            itemcode_series = df["ItemCode"].astype(str) if "ItemCode" in df.columns else pd.Series(dtype=str)
            shipto_series = df["ShipToCode"].astype(str) if "ShipToCode" in df.columns else pd.Series(dtype=str)

            st.markdown(
                f"""
                <div class="callout-banner">
                    <span class="pill-badge" style="background:#e8f0ff; color:#163ea8;">Suppliers {len(suppliers):,}</span>
                    <span class="pill-badge" style="background:#e8f7ef; color:#0f7a55;">Dynamic item filtering enabled</span>
                    Fill in the order profile below to generate a polished risk summary for a single procurement event.
                </div>
                """,
                unsafe_allow_html=True,
            )

            select_col1, select_col2, select_col3 = st.columns(3)
            with select_col1:
                purchase_name = st.selectbox("Purchase Name", suppliers, key="selected_purchase_name")

            supplier_itemcodes = sorted(
                df.loc[purchase_series == str(purchase_name), "ItemCode"].dropna().astype(str).unique().tolist()
            ) if "ItemCode" in df.columns else []

            with select_col2:
                if supplier_itemcodes:
                    item_code = st.selectbox("ItemCode", supplier_itemcodes, key="selected_item_code")
                else:
                    item_code = None
                    st.selectbox("ItemCode", ["No previous item codes for this supplier"], disabled=True)

            supplier_destinations = sorted(
                df.loc[purchase_series == str(purchase_name), "ShipToCode"].dropna().astype(str).unique().tolist()
            ) if "ShipToCode" in df.columns else []

            with select_col3:
                if supplier_destinations:
                    destination_code = st.selectbox("Destination", supplier_destinations, key="selected_shipto_code")
                else:
                    destination_code = None
                    st.selectbox("Destination", ["No historical destinations for this supplier"], disabled=True)

            if not supplier_itemcodes:
                st.warning("No historical item codes were found for the selected Purchase Name.")
            else:
                with st.form("single_prediction_form"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(
                            f"""
                            <div class="callout-banner" style="margin-top:0; margin-bottom:0.8rem;">
                                Selected supplier <strong>{purchase_name}</strong> with <strong>{len(supplier_itemcodes):,}</strong> matching historical item codes.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if destination_code is not None:
                            st.caption(f"Selected destination: {destination_code}")
                        quantity_ordered = st.number_input("Quantity Ordered", min_value=1.0, value=1.0, step=1.0)
                    with col2:
                        order_date = st.date_input("Order Date")
                        required_date = st.date_input("Required Date")
                        unit_cost = st.number_input("Unit Cost", min_value=0.0, value=0.0, step=0.01)

                    submitted = st.form_submit_button("Predict delay risk")

            if supplier_itemcodes and submitted:
                try:
                    supplier_df = df[purchase_series == str(purchase_name)].copy()
                    item_df = df[itemcode_series == str(item_code)].copy()
                    destination_df = df[shipto_series == str(destination_code)].copy() if destination_code is not None else pd.DataFrame()
                    supplier_item_df = df[
                        (purchase_series == str(purchase_name)) &
                        (itemcode_series == str(item_code))
                    ].copy()
                    supplier_destination_df = df[
                        (purchase_series == str(purchase_name)) &
                        (shipto_series == str(destination_code))
                    ].copy() if destination_code is not None else pd.DataFrame()
                    supplier_item_destination_df = df[
                        (purchase_series == str(purchase_name)) &
                        (itemcode_series == str(item_code)) &
                        (shipto_series == str(destination_code))
                    ].copy() if destination_code is not None else pd.DataFrame()

                    if not supplier_item_destination_df.empty:
                        ref_row = supplier_item_destination_df.iloc[0].copy()
                    elif not supplier_item_df.empty:
                        ref_row = supplier_item_df.iloc[0].copy()
                    elif not supplier_destination_df.empty:
                        ref_row = supplier_destination_df.iloc[0].copy()
                    elif not supplier_df.empty:
                        ref_row = supplier_df.iloc[0].copy()
                    elif not destination_df.empty:
                        ref_row = destination_df.iloc[0].copy()
                    elif not item_df.empty:
                        ref_row = item_df.iloc[0].copy()
                    else:
                        ref_row = df.iloc[0].copy()

                    order_dt = pd.to_datetime(order_date)
                    required_dt = pd.to_datetime(required_date)
                    lead_time_days = (required_dt - order_dt).days

                    if lead_time_days < 0:
                        st.error("Required Date must be after Order Date.")
                        st.stop()

                    order_value = unit_cost * quantity_ordered
                    item_order_frequency = int(len(item_df)) if not item_df.empty else 0
                    supplier_item_frequency = int(len(supplier_item_df)) if not supplier_item_df.empty else 0
                    pricing_reference_df = supplier_item_df if not supplier_item_df.empty else item_df
                    historical_item_avg_price = (
                        pricing_reference_df["UnitCost"].mean() if not pricing_reference_df.empty else np.nan
                    )
                    price_ratio = (
                        unit_cost / historical_item_avg_price
                        if pd.notna(historical_item_avg_price) and historical_item_avg_price != 0
                        else 1.0
                    )

                    input_row = {
                        "distance_km": ref_row.get("distance_km", np.nan),
                        "lead_time_days": lead_time_days,
                        "order_value": order_value,
                        "item_order_frequency": item_order_frequency,
                        "supplier_item_frequency": supplier_item_frequency,
                        "order_month": order_dt.month,
                        "order_day_of_week": order_dt.dayofweek + 1,
                        "price_ratio": price_ratio,
                        "delivery_risk_score": ref_row.get("delivery_risk_score", np.nan),
                        "weather_risk_index": ref_row.get("weather_risk_index", "unknown"),
                        "transport_region": ref_row.get("transport_region", "unknown"),
                        "transport_corridor_risk": ref_row.get("transport_corridor_risk", "unknown"),
                    }

                    if "precipitation" in st.session_state.model_results["feature_list"]:
                        input_row["precipitation"] = ref_row.get("precipitation", np.nan)
                    if "max_wind" in st.session_state.model_results["feature_list"]:
                        input_row["max_wind"] = ref_row.get("max_wind", np.nan)

                    input_df = pd.DataFrame([input_row])
                    input_df = input_df[st.session_state.model_results["feature_list"]]

                    reg_model = st.session_state.model_results["reg_model"]
                    clf_model = st.session_state.model_results["clf_model"]

                    predicted_late_days = float(reg_model.predict(input_df)[0])
                    predicted_probability = float(get_positive_class_probability(clf_model, input_df)[0])
                    classifier_label = "Late" if predicted_probability >= 0.50 else "On Time"

                    if predicted_probability >= 0.70:
                        risk_level = "High"
                    elif predicted_probability >= 0.40:
                        risk_level = "Medium"
                    else:
                        risk_level = "Low"

                    regression_signal = "Late" if predicted_late_days > 0 else "On Time"

                    risk_styles = {
                        "High": ("#fff1f2", "#b42318"),
                        "Medium": ("#fff7ed", "#b45309"),
                        "Low": ("#ecfdf3", "#0f7a55"),
                    }
                    status_styles = {
                        "Late": ("#fff1f2", "#b42318"),
                        "On Time": ("#e8f7ef", "#0f7a55"),
                    }
                    risk_bg, risk_fg = risk_styles[risk_level]
                    status_bg, status_fg = status_styles[classifier_label]

                    st.markdown(
                        f"""
                        <div class="callout-banner">
                            <span class="pill-badge" style="background:{risk_bg}; color:{risk_fg};">Risk level {risk_level}</span>
                            <span class="pill-badge" style="background:{status_bg}; color:{status_fg};">Classifier {classifier_label}</span>
                            <span class="pill-badge" style="background:#e8f0ff; color:#163ea8;">Regression {regression_signal}</span>
                            Model outputs below combine a probability estimate with a late-days forecast for a fuller operational view.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        render_result_card(
                            "Predicted late days",
                            f"{predicted_late_days:.2f}",
                            "",
                        )
                    with m2:
                        render_result_card(
                            "Delay probability",
                            f"{predicted_probability:.2%}",
                            "",
                        )
                    with m3:
                        render_result_card(
                            "Classifier status",
                            classifier_label,
                            "",
                        )
                    with m4:
                        render_result_card(
                            "Risk level",
                            risk_level,
                            "",
                        )

                    st.markdown(
                        f"""
                        <div class="callout-banner">
                            Regression signal: <strong>{regression_signal}</strong>. Classification and regression can differ because they are separate models.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    st.markdown("**Scoring inputs used**")
                    st.dataframe(input_df, use_container_width=True)

                    predicted_route = pd.DataFrame([{
                        "PurchaseName": purchase_name,
                        "ItemCode": item_code,
                        "origin_latitude": ref_row.get("origin_latitude", np.nan),
                        "origin_longitude": ref_row.get("origin_longitude", np.nan),
                        "latitude_destination": ref_row.get("latitude_destination", np.nan),
                        "longitude_destination": ref_row.get("longitude_destination", np.nan),
                        "distance_km": ref_row.get("distance_km", np.nan),
                        "predicted_late_days": predicted_late_days,
                        "delay_risk_probability": predicted_probability,
                        "delay_risk_pct": round(predicted_probability * 100, 2),
                        "predicted_status": classifier_label,
                        "risk_level": risk_level,
                        "lead_time_days": lead_time_days,
                        "order_value": order_value,
                    }])

                    st.session_state.predicted_route_df = predicted_route

                except Exception as e:
                    st.exception(e)

with tab5:
    render_section_intro(
        "Predicted Order Route Control Tower",
        "Visualize the predicted supplier-to-destination route and summarize operational risk in a control tower format.",
    )

    if st.session_state.predicted_route_df is None:
        st.info("Run a prediction in the Single Prediction tab to display its route.")
    else:
        route_df = st.session_state.predicted_route_df.copy()

        coord_cols = [
            "origin_latitude",
            "origin_longitude",
            "latitude_destination",
            "longitude_destination",
        ]

        for col in coord_cols:
            route_df[col] = pd.to_numeric(route_df[col], errors="coerce")

        numeric_cols = [
            "distance_km",
            "delay_risk_pct",
            "predicted_late_days",
            "lead_time_days",
            "order_value",
        ]
        for col in numeric_cols:
            if col in route_df.columns:
                route_df[col] = pd.to_numeric(route_df[col], errors="coerce")

        route_df = route_df.dropna(subset=coord_cols)

        if route_df.empty:
            st.warning("No valid coordinates available for the predicted route.")
        else:
            row = route_df.iloc[0]

            origin_lat = float(row["origin_latitude"])
            origin_lon = float(row["origin_longitude"])
            dest_lat = float(row["latitude_destination"])
            dest_lon = float(row["longitude_destination"])

            center_lat = (origin_lat + dest_lat) / 2
            center_lon = (origin_lon + dest_lon) / 2

            supplier = str(row.get("PurchaseName", "Unknown"))
            itemcode = str(row.get("ItemCode", "Unknown"))
            distance_km = float(row.get("distance_km", 0))
            predicted_status = str(row.get("predicted_status", "Unknown"))
            risk_level = str(row.get("risk_level", "Unknown"))
            delay_prob = float(row.get("delay_risk_pct", 0))
            predicted_late_days = float(row.get("predicted_late_days", 0))
            lead_time_days = float(row.get("lead_time_days", 0))
            order_value = float(row.get("order_value", 0))

            lat_diff = abs(origin_lat - dest_lat)
            lon_diff = abs(origin_lon - dest_lon)
            max_diff = max(lat_diff, lon_diff)

            if max_diff > 20:
                zoom_level = 3
            elif max_diff > 10:
                zoom_level = 4
            elif max_diff > 5:
                zoom_level = 5
            elif max_diff > 2:
                zoom_level = 6
            elif max_diff > 1:
                zoom_level = 7
            else:
                zoom_level = 8

            route_color = {
                "High": [220, 38, 38, 220],
                "Medium": [245, 158, 11, 220],
                "Low": [34, 197, 94, 220],
            }.get(risk_level, [59, 130, 246, 220])

            status_color = {
                "Late": "#dc2626",
                "On Time": "#16a34a",
            }.get(predicted_status, "#2563eb")

            risk_badge_bg = {
                "High": "#fee2e2",
                "Medium": "#fef3c7",
                "Low": "#dcfce7",
            }.get(risk_level, "#dbeafe")

            risk_badge_fg = {
                "High": "#991b1b",
                "Medium": "#92400e",
                "Low": "#166534",
            }.get(risk_level, "#1d4ed8")

            path_df = pd.DataFrame([{
                "path": [
                    [origin_lon, origin_lat],
                    [dest_lon, dest_lat],
                ]
            }])

            points_df = pd.DataFrame([
                {
                    "point_type": "Origin",
                    "lat": origin_lat,
                    "lon": origin_lon,
                    "label": "Origin",
                    "fill_color": [34, 197, 94, 230],
                    "line_color": [255, 255, 255, 255],
                    "radius": 18000,
                    "supplier": supplier,
                    "itemcode": itemcode,
                    "distance_km": round(distance_km, 2),
                    "predicted_status": predicted_status,
                    "risk_level": risk_level,
                    "delay_risk_pct": round(delay_prob, 2),
                },
                {
                    "point_type": "Destination",
                    "lat": dest_lat,
                    "lon": dest_lon,
                    "label": "Destination",
                    "fill_color": [30, 41, 59, 230],
                    "line_color": [255, 255, 255, 255],
                    "radius": 18000,
                    "supplier": supplier,
                    "itemcode": itemcode,
                    "distance_km": round(distance_km, 2),
                    "predicted_status": predicted_status,
                    "risk_level": risk_level,
                    "delay_risk_pct": round(delay_prob, 2),
                },
            ])

            st.markdown(
                f"""
                <style>
                .ibp-shell {{
                    background: #f8fafc;
                    border: 1px solid #dbe4ee;
                    border-radius: 18px;
                    padding: 14px 16px;
                    margin-bottom: 14px;
                    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
                }}
                .ibp-topbar {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 12px;
                    flex-wrap: wrap;
                }}
                .ibp-title {{
                    font-size: 20px;
                    font-weight: 700;
                    color: #1e3a8a;
                }}
                .ibp-sub {{
                    font-size: 12px;
                    color: #64748b;
                }}
                .ibp-chip {{
                    display: inline-block;
                    padding: 6px 10px;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 700;
                    margin-left: 8px;
                }}
                </style>

                <div class="ibp-shell">
                    <div class="ibp-topbar">
                        <div>
                            <div class="ibp-title">Procurement Visibility Control Tower</div>
                            <div class="ibp-sub">
                                Supplier route monitoring for predicted delivery risk
                            </div>
                        </div>
                        <div>
                            <span class="ibp-chip" style="background:{risk_badge_bg}; color:{risk_badge_fg};">
                                Risk: {risk_level}
                            </span>
                            <span class="ibp-chip" style="background:#e2fbe8; color:{status_color};">
                                Status: {predicted_status}
                            </span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            left_col, right_col = st.columns([1.6, 1.0], gap="large")

            with left_col:
                st.markdown("#### Route map")

                glow_layer = pdk.Layer(
                    "PathLayer",
                    data=path_df,
                    get_path="path",
                    get_width=10,
                    get_color=[255, 255, 255, 90],
                    pickable=False,
                )

                route_layer = pdk.Layer(
                    "PathLayer",
                    data=path_df,
                    get_path="path",
                    get_width=5,
                    get_color=route_color,
                    pickable=True,
                )

                marker_layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=points_df,
                    get_position='[lon, lat]',
                    get_radius="radius",
                    get_fill_color="fill_color",
                    get_line_color="line_color",
                    line_width_min_pixels=2,
                    stroked=True,
                    pickable=True,
                )

                text_layer = pdk.Layer(
                    "TextLayer",
                    data=points_df,
                    get_position='[lon, lat]',
                    get_text="label",
                    get_size=14,
                    get_color=[31, 41, 55, 255],
                    get_pixel_offset=[0, -16],
                    get_alignment_baseline="'bottom'",
                )

                deck = pdk.Deck(
                    map_style="road",
                    initial_view_state=pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=zoom_level,
                        pitch=0,
                        bearing=0,
                    ),
                    layers=[glow_layer, route_layer, marker_layer, text_layer],
                    tooltip={
                        "html": """
                            <div style="font-size:13px; padding:4px 6px;">
                                <b>{point_type}</b><br/>
                                Supplier: {supplier}<br/>
                                Item: {itemcode}<br/>
                                Distance: {distance_km} km<br/>
                                Status: {predicted_status}<br/>
                                Risk: {risk_level}<br/>
                                Delay probability: {delay_risk_pct}%
                            </div>
                        """
                    },
                )

                st.pydeck_chart(deck, use_container_width=True)

                c1, c2, c3 = st.columns(3)
                c1.metric("Origin", "Supplier")
                c2.metric("Distance", f"{distance_km:.2f} km")
                c3.metric("Destination", "Delivery site")

            with right_col:
                st.markdown("#### Alert panel")

                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        border:1px solid #dbe4ee;
                        border-radius:16px;
                        padding:14px;
                        box-shadow:0 6px 16px rgba(15,23,42,0.05);
                        margin-bottom:12px;
                    ">
                        <div style="font-size:12px; color:#64748b; margin-bottom:6px;">Supplier</div>
                        <div style="font-size:18px; font-weight:700; color:#0f172a;">{supplier}</div>
                        <div style="height:10px;"></div>
                        <div style="font-size:12px; color:#64748b; margin-bottom:6px;">Item</div>
                        <div style="font-size:16px; font-weight:600; color:#0f172a;">{itemcode}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                r1, r2 = st.columns(2)
                r1.metric("Delay probability", f"{delay_prob:.2f}%")
                r2.metric("Predicted late days", f"{predicted_late_days:.2f}")

                r3, r4 = st.columns(2)
                r3.metric("Lead time", f"{lead_time_days:.0f} days")
                r4.metric("Order value", f"${order_value:,.0f}")

                st.markdown("#### Risk interpretation")

                if delay_prob >= 70:
                    risk_note = "This order shows elevated disruption risk and should be actively monitored."
                elif delay_prob >= 40:
                    risk_note = "This order shows moderate risk and may benefit from supplier follow-up."
                else:
                    risk_note = "This order currently shows a relatively stable delivery outlook."

                st.markdown(
                    f"""
                    <div style="
                        background:#ffffff;
                        border:1px solid #dbe4ee;
                        border-radius:16px;
                        padding:14px;
                        box-shadow:0 6px 16px rgba(15,23,42,0.05);
                        margin-bottom:12px;
                    ">
                        <div style="font-size:13px; font-weight:700; color:#334155; margin-bottom:8px;">
                            Prediction summary
                        </div>
                        <div style="font-size:14px; color:#475569;">
                            {risk_note}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("#### KPI snapshot")

                kpi_df = pd.DataFrame({
                    "Metric": [
                        "Distance",
                        "Delay probability",
                        "Predicted late days",
                        "Lead time",
                        "Order value",
                    ],
                    "Value": [
                        f"{distance_km:.2f} km",
                        f"{delay_prob:.2f}%",
                        f"{predicted_late_days:.2f}",
                        f"{lead_time_days:.0f} days",
                        f"${order_value:,.0f}",
                    ]
                })
                st.dataframe(kpi_df, use_container_width=True, hide_index=True)

            st.markdown("#### Route details")

            details_df = pd.DataFrame([{
                "PurchaseName": supplier,
                "ItemCode": itemcode,
                "distance_km": round(distance_km, 3),
                "predicted_status": predicted_status,
                "risk_level": risk_level,
                "delay_risk_pct": round(delay_prob, 2),
                "predicted_late_days": round(predicted_late_days, 4),
                "lead_time_days": round(lead_time_days, 2),
                "order_value": round(order_value, 2),
                "origin_latitude": round(origin_lat, 5),
                "origin_longitude": round(origin_lon, 5),
                "latitude_destination": round(dest_lat, 5),
                "longitude_destination": round(dest_lon, 5),
            }])

            st.dataframe(details_df, use_container_width=True)
st.markdown("---")
st.caption(
    "Notes: this app combines uploaded-file feature engineering, baseline modeling, and a single-route prediction map. "
    "Weather API, reverse geocoding, and corridor shapefile logic remain lightweight placeholders for portability."
)


# %%
