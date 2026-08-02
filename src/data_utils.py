import numpy as np
import pandas as pd


def create_features(df):
    """
    Create leakage-safe forecasting features.

    Each time series is identified by:
        store_id + dept_id

    The function creates ONLY input features.

    It does NOT create the prediction target.
    Target creation is handled separately by
    create_forecast_target().

    Returns
    -------
    DataFrame
        Data with engineered forecasting features.
    """

    df = df.copy()

    # --------------------------------------------------
    # 1. Ensure correct datatype
    # --------------------------------------------------

    df["date"] = pd.to_datetime(df["date"])

    # --------------------------------------------------
    # 2. Sort chronologically within each series
    # --------------------------------------------------

    group_cols = [
        "store_id",
        "dept_id",
    ]

    df = (
        df.sort_values(
            group_cols + ["date"]
        )
        .reset_index(drop=True)
    )

    grouped_sales = df.groupby(
        group_cols,
        sort=False,
    )["sales"]

    # --------------------------------------------------
    # 3. Lag features
    # --------------------------------------------------

    lag_periods = [
        1,
        2,
        3,
        4,
        12,
        26,
        52,
    ]

    for lag in lag_periods:
        df[f"lag_{lag}"] = grouped_sales.shift(lag)

    # --------------------------------------------------
    # 4. Rolling statistics
    # --------------------------------------------------

    rolling_windows = [4, 12]

    for window in rolling_windows:

        shifted = grouped_sales.shift(1)

        df[f"rolling_mean_{window}"] = (

            shifted
            .groupby(df[group_cols].apply(tuple, axis=1))
            .transform(
                lambda s:
                    s.rolling(
                        window=window,
                        min_periods=window,
                    ).mean()
            )

        )

        df[f"rolling_std_{window}"] = (

            shifted
            .groupby(df[group_cols].apply(tuple, axis=1))
            .transform(
                lambda s:
                    s.rolling(
                        window=window,
                        min_periods=window,
                    ).std()
            )

        )

    # --------------------------------------------------
    # 5. Calendar features
    # --------------------------------------------------

    iso = df["date"].dt.isocalendar()

    df["week_of_year"] = iso.week.astype(int)

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year

    # --------------------------------------------------
    # 6. Cyclical encoding
    # --------------------------------------------------

    df["week_sin"] = np.sin(
        2 * np.pi * df["week_of_year"] / 52
    )

    df["week_cos"] = np.cos(
        2 * np.pi * df["week_of_year"] / 52
    )

    df["month_sin"] = np.sin(
        2 * np.pi * df["month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["month"] / 12
    )

    # --------------------------------------------------
    # 7. Keep rows with sufficient history
    # --------------------------------------------------

    required_history_features = (

        [f"lag_{lag}" for lag in lag_periods]

        +

        [
            "rolling_mean_4",
            "rolling_std_4",
            "rolling_mean_12",
            "rolling_std_12",
        ]

    )

    rows_before = len(df)

    df = (
        df.dropna(
            subset=required_history_features
        )
        .reset_index(drop=True)
    )

    rows_removed = rows_before - len(df)

    print(
        f"Removed {rows_removed:,} rows "
        "without sufficient historical data."
    )

    return df

def create_forecast_target(
    df: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Create a forecasting target for an arbitrary horizon.

    Parameters
    ----------
    horizon : int

        1 -> predict next week

        2 -> predict two weeks ahead

        3 -> predict three weeks ahead

        ...

    Returns
    -------
    DataFrame
        Original dataframe with new column:

            target

        where

            target = future sales
    """

    if horizon < 1:
        raise ValueError(
            "Forecast horizon must be >= 1."
        )

    df = df.copy()

    df["target"] = (
        df.groupby(
            ["store_id", "dept_id"]
        )["sales"]
        .shift(-horizon)
    )

    before = len(df)

    df = df.dropna(
        subset=["target"]
    )
    removed = before - len(df)
    print(
        f"H+{horizon}: "
        f"removed {removed:,} rows "
        f"without future targets."
    )

    return df