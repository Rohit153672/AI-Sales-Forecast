import pandas as pd

def aggregate_weekly(df):
    df = df.copy().set_index("date")

    out = []
    for (store, sku), g in df.groupby(["store_id", "sku"]):
        weekly = g["sales"].resample("W").sum().to_frame()
        weekly["price"] = g["price"].resample("W").mean()
        weekly["promo"] = g["promo"].resample("W").max().fillna(0)
        weekly = weekly.reset_index()
        weekly["store_id"] = store
        weekly["sku"] = sku
        out.append(weekly)

    return pd.concat(out, ignore_index=True)


def create_features(df):
    df = df.sort_values(["store_id", "sku", "date"]).copy()

    # Lag features
    for lag in [1, 2, 3, 4, 12]:
        df[f"lag_{lag}"] = df.groupby(["store_id", "sku"])["sales"].shift(lag)

    # Rolling mean
    df["rolling_4"] = (
        df.groupby(["store_id", "sku"])["sales"]
        .shift(1).rolling(4).mean()
    )

    # Date features
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month

    df = df.dropna()

    return df
