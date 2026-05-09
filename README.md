# Procurement Risk Modeling and Delivery Delay Prediction

## Overview

This repository contains the implementation of a procurement risk tool streamlit app developed as part of a capstone research project focused on predicting procurement delivery delays using machine learning and supplier performance analytics.

The project integrates procurement transaction data, supplier metrics, geographic information, and engineered operational features to support early-warning delivery risk prediction and proactive procurement decision-making.

The system is implemented through an interactive Streamlit application that supports:

- procurement data engineering
- supplier KPI monitoring
- delivery delay prediction
- supplier risk analysis
- procurement route visualization

The predictive framework combines feature engineering, supplier risk scoring, and machine learning techniques to estimate:

1. Whether a purchase order is likely to be delayed
2. The expected delayed days

---

# Project Objectives

The primary objectives of this project are to:

- Develop a predictive framework for procurement delivery risk
- Evaluate how supplier performance metrics improve delivery prediction accuracy
- Integrate geographic and operational procurement features into machine learning models
- Support proactive procurement decision-making through early-warning analytics
- Deploy the predictive framework through an interactive Streamlit application

---

# Repository Structure

```text
project/
│
├── data/
├── src/
├── docs/
├── requirements.txt
└── README.md
```

## Files Included

- `data/Datasets_Procurement_Cleaned_20260210_210209.xlsx` - anonymized procurement sample data.
- `data/Private_coordinates.xlsx` - anonymized private coordinate lookup used by the app.
- `src/data_cleaning_script.py` - data cleaning reference script, including raw workbook cleaning and supplier anonymization logic.
- `src/streamlit_app.py` - main Streamlit application for data upload, feature engineering, KPI dashboards, model training, single-order prediction, and route visualization.
- `src/feature_engineering.py` - export script that runs the app's procurement processing pipeline and saves the engineered modeling dataset. The feature engineering code is inside `src/streamlit_app.py`.
- `src/coordinate_privacy.py` - coordinate displacement utility used to create privacy-preserving coordinates.
- `docs/feature_engineering_weather_traffic.md` - explanation of engineered weather, transport, and corridor-risk features.
- `docs/model_comparison.md` - summary of model comparison results and interpretation.
- `requirements.txt` - Python dependencies for local use and Streamlit Community Cloud.
