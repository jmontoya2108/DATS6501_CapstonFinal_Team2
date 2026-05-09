# Procurement Risk Modeling Streamlit App

This repository contains a focused Streamlit application for procurement data engineering, supplier KPI monitoring, delay-risk modeling, single-order prediction, and supplier route visualization.

## Files Included

- `src/streamlit_app.py` - main Streamlit application for data upload, feature engineering, KPI dashboards, model training, single-order prediction, and route visualization.
- `data/Datasets_Procurement_Cleaned_20260210_210209.xlsx` - anonymized procurement sample data.
- `data/Private_coordinates.xlsx` - anonymized private coordinate lookup used by the app.
- `src/data_cleaning_script.py` - data cleaning reference script, including raw workbook cleaning and supplier anonymization logic.
- `src/feature_engineering.py` - command-line export script that runs the app's procurement processing pipeline and saves the engineered modeling dataset.
- `src/coordinate_privacy.py` - coordinate displacement utility used to create privacy-preserving coordinates.
- `docs/feature_engineering_weather_traffic.md` - explanation of engineered weather, transport, and corridor-risk features.
- `docs/model_comparison.md` - summary of model comparison results and interpretation.
- `requirements.txt` - Python dependencies for local use and Streamlit Community Cloud.

## Python Files

- `src/streamlit_app.py`: Runs the full Streamlit app. This file contains the main procurement processing function, joins procurement data with coordinate data, creates engineered features such as distance, late days, supplier risk, price signals, weather-risk proxy, and traffic/corridor-risk proxy, trains the models, and powers the dashboard, prediction, and map pages.
- `src/data_cleaning_script.py`: Documents and runs the data cleaning process for the original procurement workbook. It removes non-modeling rows, cleans key fields, handles workbook outputs, and anonymizes supplier names before the data is used in the app.
- `src/feature_engineering.py`: Provides a command-line way to generate the final engineered dataset. It calls `process_procurement()` from `src/streamlit_app.py`, then saves the output to `data/CleanandEngineered_Data.csv`.
- `src/coordinate_privacy.py`: Creates the anonymized coordinate workbook by randomly displacing original latitude and longitude values. The resulting `Private_coordinates.xlsx` file keeps map and distance features usable while protecting exact locations.
- `tmp_check_is_late.py`: Small helper script used to inspect whether `late_days` and `is_late` exist in the cleaned dataset and to calculate their basic distribution when needed.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run src/streamlit_app.py
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
