from pathlib import Path
import json

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.data_utils import create_features


# ==========================================================
# SHAP Wrapper for FastAPI
# ==========================================================

class SHAPExplain:
    """
    Wrapper around SHAP TreeExplainer
    for inference inside FastAPI.
    """

    def __init__(self, model):

        self.model = model
        self.explainer = shap.TreeExplainer(model)

    def explain(self, X):

        shap_values = self.explainer.shap_values(X)

        # Binary regression compatibility
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        return shap_values


# ==========================================================
# Project Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walmart_sales.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "lgbm.pkl"
)

FEATURE_PATH = (
    PROJECT_ROOT
    / "models"
    / "features.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)


# ==========================================================
# Explainability Report
# ==========================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    model = joblib.load(MODEL_PATH)

    # --------------------------------------------------
    # Load features
    # --------------------------------------------------

    with open(
        FEATURE_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        feature_cols = json.load(f)

    # --------------------------------------------------
    # Load data
    # --------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    df = create_features(df)

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    X = df[feature_cols]

    # ==================================================
    # FEATURE IMPORTANCE
    # ==================================================

    gain = model.feature_importance(
        importance_type="gain"
    )

    split = model.feature_importance(
        importance_type="split"
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": gain,
            "split_count": split,
        }
    )

    importance_df["gain_percent"] = (
        importance_df["gain"]
        / importance_df["gain"].sum()
        * 100
    )

    importance_df = (
        importance_df
        .sort_values(
            "gain",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\nTOP 15 FEATURES BY GAIN")
    print("=" * 70)

    print(
        importance_df[
            [
                "feature",
                "gain_percent",
                "split_count",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    # ==================================================
    # Gain Plot
    # ==================================================

    top_gain = (
        importance_df
        .head(15)
        .sort_values("gain")
    )

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_gain["feature"],
        top_gain["gain_percent"],
    )

    plt.xlabel(
        "Gain Importance (%)"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "LightGBM Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "feature_importance_gain.png",
        dpi=200,
    )

    plt.close()

    # ==================================================
    # SHAP
    # ==================================================

    sample_size = min(
        10000,
        len(X),
    )

    X_sample = X.sample(
        n=sample_size,
        random_state=42,
    )

    print(
        f"\nCalculating SHAP values for {sample_size:,} rows..."
    )

    explainer = SHAPExplain(model)

    shap_values = explainer.explain(
        X_sample
    )

    # ==================================================
    # SHAP Bar
    # ==================================================

    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_bar.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # ==================================================
    # SHAP Beeswarm
    # ==================================================

    shap.summary_plot(
        shap_values,
        X_sample,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_beeswarm.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print("\nSaved explainability outputs to:")

    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()