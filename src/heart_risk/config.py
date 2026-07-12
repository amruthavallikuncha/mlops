import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", Path.cwd())).resolve()
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "heart_disease.csv"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
PLOT_DIR = PROJECT_ROOT / "artifacts" / "plots"
MODEL_PATH = MODEL_DIR / "heart_disease_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "target"
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
RANDOM_STATE = 523
