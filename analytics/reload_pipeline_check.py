from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
pipeline = joblib.load(BASE_DIR / "best_model_pipeline.joblib")
raw_df = pd.read_csv(BASE_DIR / "titanic.csv")
raw_sample = raw_df.drop(columns=["survived", "alive"], errors="ignore").iloc[:5].copy()

predictions = pipeline.predict(raw_sample)
print("Model predictions on raw input:")
print(predictions)
print("Prediction shape:", predictions.shape)
