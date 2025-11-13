import pandas as pd
import argparse

def preprocess_walmart(train_path, stores_path, features_path, output_path):
    print("Loading datasets...")
    train = pd.read_csv(train_path)
    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)

    train['Date'] = pd.to_datetime(train['Date'])
    features['Date'] = pd.to_datetime(features['Date'])

    print("Merging datasets...")
    df = train.merge(features, on=['Store', 'Date'], how='left')
    df = df.merge(stores, on='Store', how='left')

    print("Cleaning columns...")
    df.rename(columns={
        'Store': 'store_id',
        'Dept': 'sku',
        'Weekly_Sales': 'sales'
    }, inplace=True)

    holiday_col = None
    for col in df.columns:
        if col.lower() == "isholiday":
            holiday_col = col
            break

    if holiday_col is None:
        for col in df.columns:
            if "isholiday" in col.lower():  # matches IsHoliday_x / IsHoliday_y
                holiday_col = col
                break

    if holiday_col is None:
        print("WARNING: No 'IsHoliday' column found. Setting promo = 0")
        df["promo"] = 0
    else:
        df["promo"] = df[holiday_col].astype(int)

    # Use MarkDown1 as price proxy
        df['price'] = df['MarkDown1'].fillna(0)

    df_final = df[['Date', 'store_id', 'sku', 'sales', 'price', 'promo']].copy()
    df_final.rename(columns={'Date': 'date'}, inplace=True)

    df_final = df_final.sort_values(['store_id', 'sku', 'date'])

    print(f"Saving cleaned dataset to {output_path}")
    df_final.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--stores", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", default="data/walmart_sales.csv")

    args = parser.parse_args()
    preprocess_walmart(args.train, args.stores, args.features, args.output)
