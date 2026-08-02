from pathlib import Path
import json
import random

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from src.data_utils import create_features


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "walmart_sales.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"

TUNING_RESULTS_PATH = (
    MODEL_DIR / "tuning_results.csv"
)

BEST_PARAMS_PATH = (
    MODEL_DIR / "best_params.json"
)

RANDOM_SEED = 42

N_TRIALS = 20

N_FOLDS = 3


# ============================================================
# METRIC
# ============================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


# ============================================================
# CREATE FINAL TEST SPLIT
# ============================================================

def split_development_test(
    df,
    test_ratio=0.15,
):
    """
    Reserve the final portion of unique dates as
    untouched test data.

    Everything before that becomes development data.
    """

    unique_dates = np.sort(
        df["date"].unique()
    )

    test_start_idx = int(
        len(unique_dates)
        * (1 - test_ratio)
    )

    test_cutoff = pd.Timestamp(
        unique_dates[test_start_idx]
    )

    dev_df = df[
        df["date"] < test_cutoff
    ].copy()

    test_df = df[
        df["date"] >= test_cutoff
    ].copy()

    return (
        dev_df,
        test_df,
        test_cutoff,
    )


# ============================================================
# WALK-FORWARD SPLITS
# ============================================================

def create_walk_forward_splits(
    dev_df,
    n_folds=3,
):
    """
    Expanding-window temporal validation.

    Example:

    Fold 1:
        TRAIN -------- VAL

    Fold 2:
        TRAIN ------------- VAL

    Fold 3:
        TRAIN ------------------ VAL
    """

    dates = np.sort(
        dev_df["date"].unique()
    )

    n_dates = len(dates)

    # Use roughly 20% of development dates
    # for each validation window.
    val_size = max(
        1,
        n_dates // (n_folds + 2),
    )

    initial_train_size = (
        n_dates
        - n_folds * val_size
    )

    if initial_train_size <= 0:
        raise ValueError(
            "Not enough dates for requested folds."
        )

    splits = []

    for fold in range(n_folds):

        train_end_idx = (
            initial_train_size
            + fold * val_size
        )

        val_start_idx = train_end_idx

        if fold == n_folds - 1:
            val_end_idx = n_dates

        else:
            val_end_idx = (
                val_start_idx
                + val_size
            )

        train_dates = dates[
            :train_end_idx
        ]

        val_dates = dates[
            val_start_idx:val_end_idx
        ]

        train_df = dev_df[
            dev_df["date"].isin(
                train_dates
            )
        ].copy()

        val_df = dev_df[
            dev_df["date"].isin(
                val_dates
            )
        ].copy()

        splits.append(
            (
                train_df,
                val_df,
            )
        )

    return splits


# ============================================================
# PARAMETER SAMPLING
# ============================================================

def sample_params():

    return {
        "objective": "regression",
        "metric": "rmse",

        "learning_rate": random.choice(
            [
                0.02,
                0.03,
                0.05,
                0.08,
            ]
        ),

        "num_leaves": random.choice(
            [
                15,
                31,
                63,
                127,
            ]
        ),

        "min_data_in_leaf": random.choice(
            [
                20,
                50,
                100,
                200,
            ]
        ),

        "feature_fraction": random.choice(
            [
                0.7,
                0.8,
                0.9,
                1.0,
            ]
        ),

        "bagging_fraction": random.choice(
            [
                0.7,
                0.8,
                0.9,
                1.0,
            ]
        ),

        "bagging_freq": 1,

        "lambda_l1": random.choice(
            [
                0.0,
                0.1,
                1.0,
                5.0,
            ]
        ),

        "lambda_l2": random.choice(
            [
                0.0,
                0.1,
                1.0,
                5.0,
                10.0,
            ]
        ),

        "verbosity": -1,

        "seed": RANDOM_SEED,

        "feature_fraction_seed": RANDOM_SEED,
        "bagging_seed": RANDOM_SEED,
    }


# ============================================================
# EVALUATE ONE CONFIGURATION
# ============================================================

def evaluate_params(
    params,
    splits,
    feature_cols,
):

    fold_scores = []

    best_iterations = []

    for fold_number, (
        train_df,
        val_df,
    ) in enumerate(
        splits,
        start=1,
    ):

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
                    stopping_rounds=100,
                    verbose=False,
                ),
            ],
        )

        predictions = model.predict(
            X_val,
            num_iteration=model.best_iteration,
        )

        fold_rmse = rmse(
            y_val,
            predictions,
        )

        fold_scores.append(
            fold_rmse
        )

        best_iterations.append(
            model.best_iteration
        )

        print(
            f"    Fold {fold_number}: "
            f"RMSE={fold_rmse:,.2f}, "
            f"best_iteration="
            f"{model.best_iteration}"
        )

    return {
        "mean_rmse": float(
            np.mean(fold_scores)
        ),

        "std_rmse": float(
            np.std(fold_scores)
        ),

        "mean_best_iteration": int(
            round(
                np.mean(
                    best_iterations
                )
            )
        ),

        "fold_scores": fold_scores,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    print(
        "Processed data:",
        df.shape,
    )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = create_features(
        df
    )

    df = df.sort_values(
        "date"
    ).reset_index(
        drop=True
    )

    print(
        "Feature data:",
        df.shape,
    )

    # --------------------------------------------------------
    # Reserve final test
    # --------------------------------------------------------

    (
        dev_df,
        test_df,
        test_cutoff,
    ) = split_development_test(
        df,
        test_ratio=0.15,
    )

    print("\nFINAL TEST LOCKED")
    print("=" * 60)

    print(
        "Development:",
        dev_df["date"].min().date(),
        "→",
        dev_df["date"].max().date(),
    )

    print(
        "Final test:",
        test_df["date"].min().date(),
        "→",
        test_df["date"].max().date(),
    )

    print(
        "Test cutoff:",
        test_cutoff.date(),
    )

    # --------------------------------------------------------
    # Features
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

    # --------------------------------------------------------
    # Create walk-forward folds
    # --------------------------------------------------------

    splits = create_walk_forward_splits(
        dev_df,
        n_folds=N_FOLDS,
    )

    print("\nWALK-FORWARD FOLDS")
    print("=" * 60)

    for i, (
        fold_train,
        fold_val,
    ) in enumerate(
        splits,
        start=1,
    ):

        print(
            f"\nFold {i}"
        )

        print(
            "Train:",
            fold_train[
                "date"
            ].min().date(),
            "→",
            fold_train[
                "date"
            ].max().date(),
        )

        print(
            "Validation:",
            fold_val[
                "date"
            ].min().date(),
            "→",
            fold_val[
                "date"
            ].max().date(),
        )

        print(
            "Rows:",
            len(fold_train),
            "/",
            len(fold_val),
        )

    # --------------------------------------------------------
    # Randomized parameter search
    # --------------------------------------------------------

    results = []

    seen_params = set()

    print("\nHYPERPARAMETER SEARCH")
    print("=" * 60)

    trial = 0

    while trial < N_TRIALS:

        params = sample_params()

        # Prevent accidentally evaluating the
        # exact same configuration twice.
        param_key = tuple(
            sorted(
                params.items()
            )
        )

        if param_key in seen_params:
            continue

        seen_params.add(
            param_key
        )

        trial += 1

        print(
            f"\nTrial "
            f"{trial}/{N_TRIALS}"
        )

        print(
            {
                k: v
                for k, v
                in params.items()
                if k not in {
                    "objective",
                    "metric",
                    "verbosity",
                    "seed",
                    "feature_fraction_seed",
                    "bagging_seed",
                }
            }
        )

        result = evaluate_params(
            params,
            splits,
            feature_cols,
        )

        print(
            f"    Mean RMSE: "
            f"{result['mean_rmse']:,.2f}"
        )

        print(
            f"    Std RMSE: "
            f"{result['std_rmse']:,.2f}"
        )

        row = {
            **params,

            "mean_rmse":
                result[
                    "mean_rmse"
                ],

            "std_rmse":
                result[
                    "std_rmse"
                ],

            "mean_best_iteration":
                result[
                    "mean_best_iteration"
                ],
        }

        for i, score in enumerate(
            result["fold_scores"],
            start=1,
        ):
            row[
                f"fold_{i}_rmse"
            ] = score

        results.append(
            row
        )

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            "mean_rmse"
        )
        .reset_index(
            drop=True
        )
    )

    print("\nTOP 5 CONFIGURATIONS")
    print("=" * 60)

    columns_to_show = [
        "learning_rate",
        "num_leaves",
        "min_data_in_leaf",
        "feature_fraction",
        "bagging_fraction",
        "lambda_l1",
        "lambda_l2",
        "mean_rmse",
        "std_rmse",
        "mean_best_iteration",
    ]

    print(
        results_df[
            columns_to_show
        ].head(5).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save tuning results
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        TUNING_RESULTS_PATH,
        index=False,
    )

    best_row = (
        results_df.iloc[0]
    )

    best_params = {
        "objective": "regression",
        "metric": "rmse",

        "learning_rate": float(
            best_row[
                "learning_rate"
            ]
        ),

        "num_leaves": int(
            best_row[
                "num_leaves"
            ]
        ),

        "min_data_in_leaf": int(
            best_row[
                "min_data_in_leaf"
            ]
        ),

        "feature_fraction": float(
            best_row[
                "feature_fraction"
            ]
        ),

        "bagging_fraction": float(
            best_row[
                "bagging_fraction"
            ]
        ),

        "bagging_freq": 1,

        "lambda_l1": float(
            best_row[
                "lambda_l1"
            ]
        ),

        "lambda_l2": float(
            best_row[
                "lambda_l2"
            ]
        ),

        "verbosity": -1,

        "seed": RANDOM_SEED,

        "best_num_boost_round": int(
            best_row[
                "mean_best_iteration"
            ]
        ),
    }

    with open(
        BEST_PARAMS_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            best_params,
            f,
            indent=2,
        )

    print(
        "\nTuning results saved to:",
        TUNING_RESULTS_PATH,
    )

    print(
        "Best parameters saved to:",
        BEST_PARAMS_PATH,
    )

    print(
        "\nIMPORTANT:"
        "\nThe final test set has NOT "
        "been evaluated."
    )


if __name__ == "__main__":
    main()