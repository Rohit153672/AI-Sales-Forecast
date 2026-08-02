from pathlib import Path
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from src.data_utils import create_features


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walmart_sales.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODEL_DIR / "lgbm.pkl"
FEATURE_PATH = MODEL_DIR / "features.json"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Produced by tune.py
BEST_PARAMS_PATH = MODEL_DIR / "best_params.json"


# ============================================================
# METRICS
# ============================================================

def rmse(y_true, y_pred):
    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def evaluate(y_true, y_pred):
    """
    Calculate forecasting metrics.
    """

    return {
        "rmse": rmse(
            y_true,
            y_pred,
        ),
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
    }


# ============================================================
# TEMPORAL SPLIT
# ============================================================

def temporal_split(
    df,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Chronological Train / Validation / Test split.

    We split UNIQUE DATES rather than rows.

    Approximately:

        first 70% dates  -> train
        next 15% dates   -> validation
        final 15% dates  -> test

    This ensures all observations from the same week
    belong to the same split.
    """

    unique_dates = np.sort(
        df["date"].unique()
    )

    n_dates = len(unique_dates)

    train_end_idx = int(
        n_dates * train_ratio
    )

    val_end_idx = int(
        n_dates
        * (train_ratio + val_ratio)
    )

    train_cutoff = pd.Timestamp(
        unique_dates[
            train_end_idx
        ]
    )

    val_cutoff = pd.Timestamp(
        unique_dates[
            val_end_idx
        ]
    )

    train_df = df[
        df["date"] < train_cutoff
    ].copy()

    val_df = df[
        (df["date"] >= train_cutoff)
        & (df["date"] < val_cutoff)
    ].copy()

    test_df = df[
        df["date"] >= val_cutoff
    ].copy()

    return (
        train_df,
        val_df,
        test_df,
        train_cutoff,
        val_cutoff,
    )


# ============================================================
# LOAD TUNED PARAMETERS
# ============================================================

def load_tuned_params():
    """
    Load hyperparameters selected by walk-forward
    validation in tune.py.

    best_num_boost_round is stored separately because
    we do NOT blindly use the mean CV iteration count.

    We determine the final boosting-round count using
    the recent validation window and early stopping.
    """

    if not BEST_PARAMS_PATH.exists():
        raise FileNotFoundError(
            f"\nCould not find:\n"
            f"{BEST_PARAMS_PATH}\n\n"
            "Run src/tune.py first."
        )

    with open(
        BEST_PARAMS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        tuned_config = json.load(f)

    # Don't mutate the loaded dictionary unexpectedly.
    tuned_config = tuned_config.copy()

    cv_best_iteration = tuned_config.pop(
        "best_num_boost_round",
        None,
    )

    return (
        tuned_config,
        cv_best_iteration,
    )


# ============================================================
# TRAIN WITH EARLY STOPPING
# ============================================================

def train_with_validation(
    X_train,
    y_train,
    X_val,
    y_val,
    params,
):
    """
    Train on training data while using validation data
    ONLY for early stopping.

    Final test data never enters this function.
    """

    train_data = lgb.Dataset(
        X_train,
        label=y_train,
    )

    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        reference=train_data,
    )

    model = lgb.train(
        params=params,
        train_set=train_data,

        valid_sets=[
            val_data
        ],

        valid_names=[
            "validation"
        ],

        num_boost_round=3000,

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100
            ),

            lgb.log_evaluation(
                period=50
            ),
        ],
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load processed data
    # --------------------------------------------------------

    print("\nLoading processed data...")

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    print(
        "Processed data shape:",
        df.shape,
    )

    # --------------------------------------------------------
    # 2. Feature engineering
    # --------------------------------------------------------

    print(
        "\nCreating forecasting features..."
    )

    df = create_features(
        df
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    print(
        "Feature data shape:",
        df.shape,
    )

    # --------------------------------------------------------
    # 3. Chronological Train / Validation / Test
    # --------------------------------------------------------

    (
        train_df,
        val_df,
        test_df,
        train_cutoff,
        val_cutoff,
    ) = temporal_split(
        df
    )

    print("\nTEMPORAL SPLIT")
    print("=" * 60)

    print(
        "Train:",
        train_df[
            "date"
        ].min().date(),
        "→",
        train_df[
            "date"
        ].max().date(),
    )

    print(
        "Validation:",
        val_df[
            "date"
        ].min().date(),
        "→",
        val_df[
            "date"
        ].max().date(),
    )

    print(
        "Test:",
        test_df[
            "date"
        ].min().date(),
        "→",
        test_df[
            "date"
        ].max().date(),
    )

    print()

    print(
        "Train rows:",
        len(train_df),
    )

    print(
        "Validation rows:",
        len(val_df),
    )

    print(
        "Test rows:",
        len(test_df),
    )

    # --------------------------------------------------------
    # 4. Sanity checks
    # --------------------------------------------------------

    assert (
        train_df["date"].max()
        < val_df["date"].min()
    )

    assert (
        val_df["date"].max()
        < test_df["date"].min()
    )

    print(
        "\nTemporal ordering check: PASSED"
    )

    # --------------------------------------------------------
    # 5. Define features
    # --------------------------------------------------------

    excluded_cols = {
        "date",
        "sales",
        "store_id",
        "dept_id",
        "store_type",
    }

    feature_cols = [
        col
        for col in df.columns
        if col not in excluded_cols
    ]

    print(
        "\nNumber of model features:",
        len(feature_cols),
    )

    print("\nFeatures:")

    for feature in feature_cols:
        print(
            " -",
            feature,
        )

    # --------------------------------------------------------
    # 6. Load walk-forward tuned hyperparameters
    # --------------------------------------------------------

    (
        params,
        cv_best_iteration,
    ) = load_tuned_params()

    print(
        "\nTUNED LIGHTGBM PARAMETERS"
    )

    print("=" * 60)

    for key, value in params.items():

        if key not in {
            "objective",
            "metric",
            "verbosity",
            "seed",
            "feature_fraction_seed",
            "bagging_seed",
        }:

            print(
                f"{key}: {value}"
            )

    if cv_best_iteration is not None:

        print(
            "\nMean best iteration "
            "from walk-forward CV:",
            cv_best_iteration,
        )

    # --------------------------------------------------------
    # 7. Training matrices
    # --------------------------------------------------------

    X_train = train_df[
        feature_cols
    ]

    y_train = train_df[
        "sales"
    ]

    X_val = val_df[
        feature_cols
    ]

    y_val = val_df[
        "sales"
    ]

    # --------------------------------------------------------
    # 8. Validation baseline
    # --------------------------------------------------------

    val_baseline_predictions = (
        val_df["lag_1"]
    )

    val_baseline_metrics = evaluate(
        y_val,
        val_baseline_predictions,
    )

    # --------------------------------------------------------
    # 9. Train tuned model using validation
    # --------------------------------------------------------

    print(
        "\nTraining tuned LightGBM..."
    )

    model = train_with_validation(
        X_train,
        y_train,
        X_val,
        y_val,
        params,
    )

    best_iteration = (
        model.best_iteration
    )

    print(
        "\nBest iteration selected "
        "using validation:",
        best_iteration,
    )

    # --------------------------------------------------------
    # 10. Validation predictions
    # --------------------------------------------------------

    val_predictions = model.predict(
        X_val,
        num_iteration=best_iteration,
    )

    val_metrics = evaluate(
        y_val,
        val_predictions,
    )

    # --------------------------------------------------------
    # 11. Validation improvement
    # --------------------------------------------------------

    val_rmse_improvement = (
        (
            val_baseline_metrics["rmse"]
            - val_metrics["rmse"]
        )
        / val_baseline_metrics["rmse"]
        * 100
    )

    val_mae_improvement = (
        (
            val_baseline_metrics["mae"]
            - val_metrics["mae"]
        )
        / val_baseline_metrics["mae"]
        * 100
    )

    print(
        "\nVALIDATION RESULTS"
    )

    print("=" * 60)

    print(
        "\nLag-1 baseline"
    )

    print(
        f"RMSE: "
        f"{val_baseline_metrics['rmse']:,.2f}"
    )

    print(
        f"MAE:  "
        f"{val_baseline_metrics['mae']:,.2f}"
    )

    print(
        "\nTuned LightGBM"
    )

    print(
        f"RMSE: "
        f"{val_metrics['rmse']:,.2f}"
    )

    print(
        f"MAE:  "
        f"{val_metrics['mae']:,.2f}"
    )

    print(
        "\nValidation improvement"
    )

    print(
        f"RMSE reduction: "
        f"{val_rmse_improvement:.2f}%"
    )

    print(
        f"MAE reduction: "
        f"{val_mae_improvement:.2f}%"
    )

    # --------------------------------------------------------
    # 12. Combine Train + Validation
    # --------------------------------------------------------

    print(
        "\nCombining TRAIN + VALIDATION..."
    )

    train_val_df = pd.concat(
        [
            train_df,
            val_df,
        ],
        ignore_index=True,
    )

    train_val_df = (
        train_val_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    X_train_val = train_val_df[
        feature_cols
    ]

    y_train_val = train_val_df[
        "sales"
    ]

    print(
        "Final training rows:",
        len(train_val_df),
    )

    print(
        "Final training period:",
        train_val_df[
            "date"
        ].min().date(),
        "→",
        train_val_df[
            "date"
        ].max().date(),
    )

    # --------------------------------------------------------
    # 13. Retrain FINAL model
    # --------------------------------------------------------

    final_train_data = lgb.Dataset(
        X_train_val,
        label=y_train_val,
    )

    print(
        "\nRetraining final model "
        f"for {best_iteration} boosting rounds..."
    )

    final_model = lgb.train(
        params=params,
        train_set=final_train_data,
        num_boost_round=best_iteration,
    )

    # --------------------------------------------------------
    # 14. FINAL TEST
    # --------------------------------------------------------
    #
    # This is the first point where test performance
    # is evaluated.
    # --------------------------------------------------------

    print(
        "\nEvaluating locked final test..."
    )

    X_test = test_df[
        feature_cols
    ]

    y_test = test_df[
        "sales"
    ]

    test_predictions = (
        final_model.predict(
            X_test
        )
    )

    test_metrics = evaluate(
        y_test,
        test_predictions,
    )

    # --------------------------------------------------------
    # 15. Test lag-1 baseline
    # --------------------------------------------------------

    test_baseline_predictions = (
        test_df["lag_1"]
    )

    test_baseline_metrics = evaluate(
        y_test,
        test_baseline_predictions,
    )

    seasonal_baseline_predictions = test_df["lag_52"]

    seasonal_baseline_metrics = evaluate(
        y_test,
        seasonal_baseline_predictions,
    )

    # --------------------------------------------------------
    # 16. Test improvement
    # --------------------------------------------------------

    test_rmse_improvement = (
        (
            test_baseline_metrics["rmse"]
            - test_metrics["rmse"]
        )
        / test_baseline_metrics["rmse"]
        * 100
    )

    test_mae_improvement = (
        (
            test_baseline_metrics["mae"]
            - test_metrics["mae"]
        )
        / test_baseline_metrics["mae"]
        * 100
    )

    # --------------------------------------------------------
    # 17. FINAL RESULTS
    # --------------------------------------------------------

    print(
        "\nFINAL TEST RESULTS"
    )


    print(
        "\nNaive lag-1 baseline"
    )

    print(
        f"RMSE: "
        f"{test_baseline_metrics['rmse']:,.2f}"
    )

    print(
        f"MAE:  "
        f"{test_baseline_metrics['mae']:,.2f}"
    )

    print(
        "\nTuned LightGBM"
    )

    print(
        f"RMSE: "
        f"{test_metrics['rmse']:,.2f}"
    )

    print(
        f"MAE:  "
        f"{test_metrics['mae']:,.2f}"
    )

    print(
        "\nImprovement over lag-1 baseline"
    )

    print(
        f"RMSE reduction: "
        f"{test_rmse_improvement:.2f}%"
    )

    print(
        f"MAE reduction: "
        f"{test_mae_improvement:.2f}%"
    )

    print(
    "\nSeasonal lag-52 baseline"
)

    print(
        f"RMSE: "
        f"{seasonal_baseline_metrics['rmse']:,.2f}"
    )

    print(
        f"MAE:  "
        f"{seasonal_baseline_metrics['mae']:,.2f}"
    )

    # --------------------------------------------------------
    # 18. Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        final_model,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # 19. Save feature schema
    # --------------------------------------------------------

    with open(
        FEATURE_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            feature_cols,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 20. Save metrics + experiment information
    # --------------------------------------------------------

    metrics = {

        # -------------------------
        # Data periods
        # -------------------------

        "train_start": str(
            train_df[
                "date"
            ].min().date()
        ),

        "train_end": str(
            train_df[
                "date"
            ].max().date()
        ),

        "validation_start": str(
            val_df[
                "date"
            ].min().date()
        ),

        "validation_end": str(
            val_df[
                "date"
            ].max().date()
        ),

        "test_start": str(
            test_df[
                "date"
            ].min().date()
        ),

        "test_end": str(
            test_df[
                "date"
            ].max().date()
        ),

        # -------------------------
        # Dataset sizes
        # -------------------------

        "train_rows": int(
            len(train_df)
        ),

        "validation_rows": int(
            len(val_df)
        ),

        "test_rows": int(
            len(test_df)
        ),

        "final_training_rows": int(
            len(train_val_df)
        ),

        # -------------------------
        # Boosting rounds
        # -------------------------

        "cv_mean_best_iteration": (
            int(cv_best_iteration)
            if cv_best_iteration is not None
            else None
        ),

        "validation_selected_iteration": int(
            best_iteration
        ),

        # -------------------------
        # Validation metrics
        # -------------------------

        "validation_baseline_rmse": float(
            val_baseline_metrics["rmse"]
        ),

        "validation_lightgbm_rmse": float(
            val_metrics["rmse"]
        ),

        "validation_baseline_mae": float(
            val_baseline_metrics["mae"]
        ),

        "validation_lightgbm_mae": float(
            val_metrics["mae"]
        ),

        "validation_rmse_improvement_percent": float(
            val_rmse_improvement
        ),

        "validation_mae_improvement_percent": float(
            val_mae_improvement
        ),

        # -------------------------
        # FINAL test metrics
        # -------------------------

        "test_baseline_rmse": float(
            test_baseline_metrics["rmse"]
        ),

        "test_lightgbm_rmse": float(
            test_metrics["rmse"]
        ),

        "test_baseline_mae": float(
            test_baseline_metrics["mae"]
        ),

        "test_lightgbm_mae": float(
            test_metrics["mae"]
        ),

        "test_rmse_improvement_percent": float(
            test_rmse_improvement
        ),

        "test_mae_improvement_percent": float(
            test_mae_improvement
        ),

        # -------------------------
        # Hyperparameters
        # -------------------------

        "hyperparameters": params,
    }

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # 21. Finished
    # --------------------------------------------------------

    print(
        "\nARTIFACTS SAVED"
    )

    print("=" * 60)

    print(
        "Model:",
        MODEL_PATH,
    )

    print(
        "Features:",
        FEATURE_PATH,
    )

    print(
        "Metrics:",
        METRICS_PATH,
    )


if __name__ == "__main__":
    main()