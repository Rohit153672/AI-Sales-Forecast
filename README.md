# 📦 AI Supply Chain Forecasting & Inventory Intelligence

An end-to-end machine learning application that predicts weekly retail demand and provides inventory recommendations using LightGBM, FastAPI, and Streamlit.

---

## Overview

This project forecasts weekly product demand from historical sales and business data. It also calculates inventory metrics such as Safety Stock and Reorder Point, while explaining each prediction using SHAP.

The application includes a FastAPI backend for real-time predictions and a Streamlit dashboard for interactive visualization.

---

## Key Features

- 📈 Weekly demand forecasting using LightGBM
- ⚙️ Time-series feature engineering with historical sales and calendar information
- 🔍 SHAP explainability for prediction insights
- 📦 Inventory planning using Safety Stock and Reorder Point
- 🚀 FastAPI REST API for real-time inference
- 💻 Interactive Streamlit dashboard with Plotly visualizations

---

## Tech Stack

- **Language:** Python
- **Machine Learning:** LightGBM, Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Explainability:** SHAP
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Visualization:** Plotly

---

## Results

Compared with a simple lag-based forecasting approach, the model achieved:

- ✅ **24.4% lower RMSE**
- ✅ **17.8% lower MAE**

through feature engineering and hyperparameter tuning.

---

## Dashboard

The dashboard allows users to:

- Generate demand forecasts
- View historical sales trends
- Monitor inventory metrics
- Understand model predictions using SHAP

> Add dashboard screenshots here.

---

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `/forecast` | Predict weekly demand |
| `/history` | Retrieve historical sales |
| `/inventory` | Calculate inventory metrics |
| `/explain` | Explain predictions using SHAP |

---

## Project Structure

```
AI-Sales-Forecast/
│
├── app.py
├── src/
├── models/
├── data/
├── reports/
└── requirements.txt
```

---

## Run Locally

```bash
pip install -r requirements.txt

uvicorn src.api:app --reload

streamlit run app.py
```

---

## Future Improvements

- Multi-step forecasting
- Cloud deployment
- Docker support
- MLOps pipeline
