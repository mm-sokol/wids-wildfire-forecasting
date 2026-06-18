"""
Code to run model inference with trained models
"""

import json
from pathlib import Path
import pickle

from config import MODELS_DIR, PROCESSED_DATA_DIR, seed_everything, SEED
from features import FeatureSelectionPipeline, FilterByL1, FilterCorrelated
from modeling.models import HORIZONS
import numpy as np
import pandas as pd
from logging import getLogger, basicConfig, INFO
from sksurv.metrics import concordance_index_censored, integrated_brier_score, brier_score
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

basicConfig(level=INFO)
logger = getLogger(__name__)

def read_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_clean.csv",
    feature_subset: Path|None = PROCESSED_DATA_DIR / "features.json",
    model_path: Path = MODELS_DIR / "rsf_model.pkl",
    on_full_test: bool = False,
):
    seed_everything(SEED)
    
    # Load data
    X = pd.read_csv(features_path)
    
    if on_full_test:
        logger.info("Testing on full test data")
        X_test = X

    else:
        logger.info("Testing on train/test split")
        y = X[["time_to_hit_hours", "event"]]
        X = X.drop(columns=["time_to_hit_hours", "event"])
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=0.2, 
            random_state=SEED,
            stratify=y['event']
        )
        y_structured = Surv.from_dataframe(
            event='event', 
            time='time_to_hit_hours', 
            data=y_test.reset_index(drop=True)
        )
        
    if feature_subset is not None:
        if not feature_subset.is_file():
            logger.error("No valid feature set json file was provided")
        with open(feature_subset, "r") as feat:
            feature_columns = json.load(feat)
            logger.info(f"Feature columns: {feature_columns}")
        
        # Predict using the loaded feature columns
        X_test = X_test[feature_columns]
        
        
    model = read_model(model_path)

    predicted_probabilities = model.predict(X_test)
    safe_max = y_test['time_to_hit_hours'].max() - 1e-5
    horizons_array = np.array(HORIZONS)
    time_horizons = horizons_array[horizons_array < safe_max]
    _, brier_scores = brier_score(y_train, y_structured, predicted_probabilities, time_horizons)

    for horizon, score in zip(HORIZONS, brier_scores):
        print(f"Brier Score at horizon {horizon}: {score:.4f}")
        
    


if __name__ == "__main__":

    main(
        features_path = PROCESSED_DATA_DIR / "train_clean.csv",
        feature_subset = MODELS_DIR / "train_v1_2026-06-18_08-47-17" / "features.json",
        model_path = MODELS_DIR / "train_v1_2026-06-18_08-47-17" / "rsf_model.pkl",
        on_full_test = False,
    )
