from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from src.explain import SHAPExplain
from src.supply_chain import calculate_inventory

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model, features, explainer

    data = joblib.load("models/lgbm.pkl")
    model = data["model"]
    features = data["features"]
    explainer = SHAPExplain(model)


class PredictRequest(BaseModel):
    data: dict


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.data])

        pred = float(model.predict(df[features])[0])

        # SHAP
        shap_vals = explainer.explain(df[features])[0]

        
        drivers = {
            features[i]: float(np.nan_to_num(shap_vals[i]))
            for i in range(len(features))
        }

        top_5 = dict(sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

        # Inventory
        demand_series = pd.Series(np.random.normal(pred, pred*0.1, 10))
        inventory = calculate_inventory(demand_series)

        return {
            "forecast": float(pred),
            "drivers": top_5,
            "inventory": inventory
        }

    except Exception as e:
        return {"error": str(e)}
    i

