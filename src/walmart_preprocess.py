from pathlib import Path
import pandas as pd
import os

print("Current working directory:", os.getcwd())


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def preprocess_walmart(
    train_path,
    stores_path,
    features_path,
    output_path,
):
    """
    Prepare the Walmart Store Sales dataset for forecasting.

    Forecasting granularity:
        store_id + dept_id + week

    Target:
        sales (originally Weekly_Sales)
    """
 # Load data


    train = pd.read_csv(train_path)
    stores = pd.read_csv(stores_path)
    features = pd.read_csv(features_path)

    print("Train shape:", train.shape)
    print("Stores shape:", stores.shape)
    print("Features shape:", features.shape)

    # --------------------------------------------------
    # 2. Convert dates
    # --------------------------------------------------

    train["Date"] = pd.to_datetime(train["Date"])
    features["Date"] = pd.to_datetime(features["Date"])

    # --------------------------------------------------
    # 3. Validate merge keys
    # --------------------------------------------------

    if features.duplicated(["Store", "Date"]).any():
        raise ValueError(
            "features.csv contains duplicate Store-Date rows."
        )

    if stores.duplicated(["Store"]).any():
        raise ValueError(
            "stores.csv contains duplicate Store rows."
        )

    # --------------------------------------------------
    # 4. Merge contextual data
    # --------------------------------------------------

    rows_before = len(train)

    df = train.merge(
        features,
        on=["Store", "Date"],
        how="left",
        suffixes=("_train", "_feature"),
        validate="many_to_one",
    )

    df = df.merge(
        stores,
        on="Store",
        how="left",
        validate="many_to_one",
    )

    if len(df) != rows_before:
        raise ValueError(
            "Unexpected row-count change after merging."
        )

    # --------------------------------------------------
    # 5. Resolve holiday column
    # --------------------------------------------------

    # Both train and features can contain IsHoliday.
    if "IsHoliday_train" in df.columns:
        df["is_holiday"] = (
            df["IsHoliday_train"]
            .fillna(False)
            .astype(int)
        )

    elif "IsHoliday" in df.columns:
        df["is_holiday"] = (
            df["IsHoliday"]
            .fillna(False)
            .astype(int)
        )

    else:
        raise ValueError(
            "Could not find IsHoliday after merge."
        )

    # --------------------------------------------------
    # 6. Rename fields truthfully
    # --------------------------------------------------

    df = df.rename(
        columns={
            "Date": "date",
            "Store": "store_id",
            "Dept": "dept_id",
            "Weekly_Sales": "sales",
            "Type": "store_type",
            "Size": "store_size",
            "Temperature": "temperature",
            "Fuel_Price": "fuel_price",
            "CPI": "cpi",
            "Unemployment": "unemployment",
            "MarkDown1": "markdown_1",
            "MarkDown2": "markdown_2",
            "MarkDown3": "markdown_3",
            "MarkDown4": "markdown_4",
            "MarkDown5": "markdown_5",
        }
    )

    # --------------------------------------------------
    # 7. Select model-relevant columns
    # --------------------------------------------------

    columns = [
        "date",
        "store_id",
        "dept_id",
        "sales",
        "is_holiday",
        "store_type",
        "store_size",
        "temperature",
        "fuel_price",
        "cpi",
        "unemployment",
        "markdown_1",
        "markdown_2",
        "markdown_3",
        "markdown_4",
        "markdown_5",
    ]

    missing_columns = [
        col for col in columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    df = df[columns].copy()

    # --------------------------------------------------
    # 8. Missing markdowns
    # --------------------------------------------------

    markdown_cols = [
        "markdown_1",
        "markdown_2",
        "markdown_3",
        "markdown_4",
        "markdown_5",
    ]

    df[markdown_cols] = df[markdown_cols].fillna(0)

    # --------------------------------------------------
    # 9. Sort chronologically within each series
    # --------------------------------------------------

    df = df.sort_values(
        ["store_id", "dept_id", "date"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 10. Basic validation
    # --------------------------------------------------

    duplicate_rows = df.duplicated(
        ["store_id", "dept_id", "date"]
    ).sum()

    if duplicate_rows:
        raise ValueError(
            f"Found {duplicate_rows} duplicate "
            "store-department-date observations."
        )

    print("\nProcessed shape:", df.shape)
    print("\nMissing values:")
    print(df.isna().sum())

    # --------------------------------------------------
    # 11. Save
    # --------------------------------------------------

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"\nSaved processed data to: {output_path}")

    return df


if __name__ == "__main__":
    preprocess_walmart(
        train_path=RAW_DATA_DIR / "train.csv",
        stores_path=RAW_DATA_DIR / "stores.csv",
        features_path=RAW_DATA_DIR / "features.csv",
        output_path=PROCESSED_DATA_DIR / "walmart_sales.csv",
    )