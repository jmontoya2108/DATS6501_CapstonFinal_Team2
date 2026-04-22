# Procurement Risk Modeling Streamlit App

This repository contains a focused Streamlit application for procurement data engineering, supplier KPI monitoring, delay-risk modeling, single-order prediction, and supplier route visualization.

## Files Included

- `streamlit_app.py` - main Streamlit application.
- `data/Datasets_Procurement_Cleaned_20260210_210209.xlsx` - anonymized procurement sample data.
- `data/Private_coordinates.xlsx` - anonymized private coordinate lookup used by the app.
- `data_cleaning_script.py` - data cleaning reference script, including supplier anonymization logic.
- `src/feature_engineering.py` - command-line export of the engineered modeling dataset.
- `src/coordinate_privacy.py` - coordinate displacement utility used to create privacy-preserving coordinates.
- `docs/feature_engineering_weather_traffic.md` - explanation of engineered weather, transport, and corridor-risk features.
- `assets/` - feature engineering and model pipeline diagrams.
- `requirements.txt` - Python dependencies for local use and Streamlit Community Cloud.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## App Workflow

1. Open the app.
2. Use the bundled sample files or upload procurement and coordinate Excel files.
3. Run data engineering.
4. Review engineered data and KPIs.
5. Train models before using the single prediction and map views.

## Privacy Note

Supplier names in the included sample files have been anonymized as `Supplier 1`, `Supplier 2`, `Supplier 3`, and so on. The coordinate lookup has been anonymized with the same mapping so the app joins correctly.

## Feature Engineering Notes

The Streamlit app generates route, timing, supplier-history, pricing, weather-risk, and traffic/corridor proxy features during the data engineering step. See [docs/feature_engineering_weather_traffic.md](docs/feature_engineering_weather_traffic.md) for details.
