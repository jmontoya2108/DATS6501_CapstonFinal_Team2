# Export script that runs the app's procurement processing pipeline and saves the engineered modeling dataset.
# The main feature engineering logic is in src/streamlit_app.py.
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from streamlit_app import get_default_data_sources, process_procurement


OUTPUT_CSV = PROJECT_ROOT / "data" / "CleanandEngineered_Data.csv"


def main():
    procurement_path, coords_path = get_default_data_sources()
    if procurement_path is None or coords_path is None:
        raise FileNotFoundError("Default procurement or coordinate source files were not found.")

    print(f"Procurement source: {procurement_path}")
    print(f"Coordinates source: {coords_path}")

    df_final = process_procurement(procurement_path, coords_path)
    df_final.to_csv(OUTPUT_CSV, index=False)

    supplier_cols = [
        "PurchaseName",
        "late_rate",
        "extension_rate",
        "lateness_severity_score",
        "delivery_risk_score",
    ]

    print(f"Saved engineered data to: {OUTPUT_CSV}")
    print(f"Shape: {df_final.shape}")
    print("New supplier delivery risk columns:")
    print(df_final[supplier_cols].drop_duplicates().head(10).to_string(index=False))


if __name__ == "__main__":
    main()
