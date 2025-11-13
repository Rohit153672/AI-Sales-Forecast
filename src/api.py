from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from src.explain import SHAPExplain

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
    df = pd.DataFrame([req.data])
    pred = float(model.predict(df[features])[0])

    shap_vals = explainer.explain(df[features])[0]
    drivers = {
        features[i]: float(shap_vals[i])
        for i in range(len(features))
    }

    top_5 = dict(sorted(drivers.items(), key=lambda x: abs(x[1]), reverse=True)[:5])

    return {
        "forecast": pred,
        "drivers": top_5
    }
