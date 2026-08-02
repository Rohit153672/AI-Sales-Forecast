from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import HTTPException

from src.explain import SHAPExplain
from src.supply_chain import calculate_inventory
from src.feature_builder import FeatureBuilder

builder = None

# FastAPI

app = FastAPI(
    title="Supply Chain Forecast API",
    description="Enterprise demand forecasting and inventory optimization",
    version="1.0.0",
)
# Globals

model = None
features = None
explainer = None

# Startup

model = None
features = None
builder = None
explainer = None


@app.on_event("startup")
def startup():

    global model
    global features
    global builder
    global explainer

    model = joblib.load("models/lgbm.pkl")

    with open("models/features.json") as f:
        features = json.load(f)

    builder = FeatureBuilder(
        "data/processed/walmart_sales.csv"
    )

    explainer = SHAPExplain(model)

    print("Model loaded successfully.")

# Request Model

class ForecastRequest(BaseModel):
    store_id: int
    dept_id: int
    forecast_date: str

# Utility

def validate_features(df):

    missing = [
        col
        for col in features
        if col not in df.columns
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing features: {missing}"
        )

# Home

@app.get("/")
def home():

    return {
        "message":"Supply Chain Forecast API",
        "model":"LightGBM",
        "status":"running"
    }

# Health

@app.get("/health")
def health():

    return {
        "status":"healthy"
    }

# Forecast Endpoint

@app.post("/forecast")
def forecast(req: ForecastRequest):

    try:
        X = builder.build_feature_vector(
            req.store_id,
            req.dept_id,
            req.forecast_date,
        )

        prediction = float(
            model.predict(X[features])[0]
        )

        return {
            "forecast": prediction,
            "unit": "weekly_sales",
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

# --------------------------------------------------------
# Explain Endpoint
# --------------------------------------------------------

@app.post("/explain")
def explain(req: ForecastRequest):

    try:

        df = builder.build_feature_vector(req.store_id,req.dept_id,req.forecast_date,)

        validate_features(df)

        shap_values = explainer.explain(
            df[features]
        )[0]

        drivers = []

        for i, feature in enumerate(features):

            drivers.append({

                "feature": feature,

                "impact": float(
                    np.nan_to_num(
                        shap_values[i]
                    )
                )

            })

        drivers = sorted(

            drivers,

            key=lambda x: abs(
                x["impact"]
            ),

            reverse=True,

        )[:5]

        return {

            "drivers": drivers

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

# --------------------------------------------------------
# Inventory Endpoint
# --------------------------------------------------------

@app.post("/inventory")
def inventory(req: ForecastRequest):

    try:
        df = builder.build_feature_vector(req.store_id,req.dept_id,req.forecast_date,)
        validate_features(df)

        prediction = float(
            model.predict(
                df[features]
            )[0]
        )
        history = [
        df["lag_4"].iloc[0],
        df["lag_3"].iloc[0],
        df["lag_2"].iloc[0],
        df["lag_1"].iloc[0],
        prediction,]

        demand = pd.Series(history)

        inventory_plan = calculate_inventory(
            demand
        )

        return {
            "forecast": prediction,
            "inventory": inventory_plan
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/history")
def history(req: ForecastRequest):

    try:

        history = builder.get_history(
            store_id=req.store_id,
            dept_id=req.dept_id,
            forecast_date=req.forecast_date,
        )

        return {
            "dates": history["date"].dt.strftime("%Y-%m-%d").tolist(),
            "sales": history["sales"].tolist(),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

@app.post("/explain")
def explain(req: ForecastRequest):

    try:

        X = builder.build_feature_vector(
            req.store_id,
            req.dept_id,
            req.forecast_date,
        )

        shap_values = explainer.explain(X)[0]

        drivers = []

        for feature, value in zip(features, shap_values):

            drivers.append({
                "feature": feature,
                "impact": float(value)
            })

        drivers = sorted(
            drivers,
            key=lambda x: abs(x["impact"]),
            reverse=True,
        )

        return {
            "drivers": drivers[:10]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )