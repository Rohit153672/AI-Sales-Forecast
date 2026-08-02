from pathlib import Path
import pandas as pd

from src.data_utils import create_features
from trainer import train_one_horizon

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------
# Load processed dataset
# ---------------------------------------------------------

print("Loading processed data...")

df = pd.read_csv(
    PROJECT_ROOT / "data" / "processed" / "walmart_sales.csv",
    parse_dates=["date"],
)

print("Creating features...")

df = create_features(df)

# ---------------------------------------------------------
# Train H+1 model
# ---------------------------------------------------------

metrics = train_one_horizon(
    df=df,
    horizon=1,
    model_dir=MODEL_DIR,
)

print("\nReturned metrics")

for k, v in metrics.items():
    print(f"{k}: {v}")