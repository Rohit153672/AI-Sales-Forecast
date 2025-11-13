import argparse
import joblib
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from data_utils import aggregate_weekly, create_features

def train_model(data_path, output_path):
    print("Loading dataset...")
    df = pd.read_csv(data_path, parse_dates=["date"])

    print("Aggregating weekly...")
    df = aggregate_weekly(df)

    print("Creating features...")
    df = create_features(df)

    feature_cols = [c for c in df.columns if c not in ["date", "sales", "store_id", "sku"]]

    X = df[feature_cols]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1
    }

    print("Training LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=500)
    
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    print(f"Validation RMSE: {rmse:.4f}")

   

    joblib.dump({"model": model, "features": feature_cols}, output_path)
    print(f"Saved model → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/walmart_sales.csv")
    parser.add_argument("--output", default="models/lgbm.pkl")
    args = parser.parse_args()

    train_model(args.data, args.output)
