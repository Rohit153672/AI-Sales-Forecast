from pathlib import Path

import numpy as np
import pandas as pd

class FeatureBuilder:
    """
    Automatically generates model features
    from historical Walmart sales.
    """
    def __init__(self, history_path):
        self.history = pd.read_csv(history_path,parse_dates=["date"],)
        self.history = self.history.sort_values(
            ["store_id","dept_id","date",]).reset_index(drop=True)


    def get_history(self,store_id,dept_id,forecast_date,):
        """
        Return all observations before
        the requested forecast date.
        """

        history = self.history[
            (self.history["store_id"] == store_id) & (self.history["dept_id"] == dept_id) & (self.history["date"] < forecast_date)
        ].copy()

        history = history.sort_values("date")
        return history

    def validate_history(self,history,):
        """
        Ensure enough observations exist
        to create lag and rolling features.
        """
        minimum_history = 52
        if len(history) < minimum_history:
            raise ValueError(
                f"Need at least {minimum_history} "
                f"weeks of history."
            )

    def build_feature_vector(
    self,
    store_id,
    dept_id,
    forecast_date,
    ):
        """
        Build one feature vector exactly like the
        training dataset.
        """

        forecast_date = pd.to_datetime(forecast_date)

        history = self.get_history(
            store_id,
            dept_id,
            forecast_date,
        )

        self.validate_history(history)

        latest = history.iloc[-1]

        features = {}

        # --------------------------------------------------
        # Static / External Features
        # --------------------------------------------------

        features["is_holiday"] = latest["is_holiday"]
        features["store_size"] = latest["store_size"]

        features["temperature"] = latest["temperature"]
        features["fuel_price"] = latest["fuel_price"]
        features["cpi"] = latest["cpi"]
        features["unemployment"] = latest["unemployment"]

        features["markdown_1"] = latest["markdown_1"]
        features["markdown_2"] = latest["markdown_2"]
        features["markdown_3"] = latest["markdown_3"]
        features["markdown_4"] = latest["markdown_4"]
        features["markdown_5"] = latest["markdown_5"]

        # --------------------------------------------------
        # Lag Features
        # --------------------------------------------------

        sales = history["sales"].values

        features["lag_1"] = sales[-1]
        features["lag_2"] = sales[-2]
        features["lag_3"] = sales[-3]
        features["lag_4"] = sales[-4]

        features["lag_12"] = sales[-12]
        features["lag_26"] = sales[-26]
        features["lag_52"] = sales[-52]

        # --------------------------------------------------
        # Rolling Features
        # --------------------------------------------------

        features["rolling_mean_4"] = np.mean(sales[-4:])
        features["rolling_std_4"] = np.std(sales[-4:])

        features["rolling_mean_12"] = np.mean(sales[-12:])
        features["rolling_std_12"] = np.std(sales[-12:])

        # --------------------------------------------------
        # Calendar Features
        # --------------------------------------------------

        week = forecast_date.isocalendar().week
        month = forecast_date.month
        quarter = forecast_date.quarter
        year = forecast_date.year

        features["week_of_year"] = int(week)
        features["month"] = month
        features["quarter"] = quarter
        features["year"] = year

        features["week_sin"] = np.sin(
            2 * np.pi * week / 52
        )

        features["week_cos"] = np.cos(
            2 * np.pi * week / 52
        )

        features["month_sin"] = np.sin(
            2 * np.pi * month / 12
        )

        features["month_cos"] = np.cos(
            2 * np.pi * month / 12
        )

        return pd.DataFrame([features])