"""
Code to train models
"""

from logging import getLogger, basicConfig, INFO
from pathlib import Path

from config import MODELS_DIR, PROCESSED_DATA_DIR, seed_everything, SEED
from modeling.models import build_models

import pandas as pd
import numpy as np
import json

from sklearn.model_selection import KFold, train_test_split
from features import FilterCorrelated, FilterByL1, FeatureSelectionPipeline
from sksurv.util import Surv

basicConfig(level=INFO)
logger = getLogger(__name__)

def read_hyperparameters(path: Path) -> dict:
    with open(path, "r") as f:
        logger.info(f"Loaded hyperparameters from {path}")
        return json.load(f)
    logger.warning(f"Could not load hyperparameters from {path}, using empty dict")
    return {}

def save_feature_columns(columns: list, path: Path):
    with open(path, "w") as f:
        json.dump(columns, f, indent=2)
    logger.info(f"Saved feature columns to {path}")

def main(
    model_name: str = "rsf",
    features_path: Path = PROCESSED_DATA_DIR / "train_clean.csv",
    hparams_path: Path = MODELS_DIR / "rsf_best_params.json",
    model_path: Path = MODELS_DIR / "rsf_model.pkl",
    on_full_train: bool = False,
):
    seed_everything(SEED)
    
    model_path.parent.mkdir(exist_ok=True)
    
    # Load data
    X = pd.read_csv(features_path)
    y = X[["time_to_hit_hours", "event"]]
    X = X.drop(columns=["time_to_hit_hours", "event"])
    
    if on_full_train:
        logger.info("Training on full training data")
        X_train = X
        y_train = y
    else:
        logger.info("Training on train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=0.2, 
            random_state=SEED,
            stratify=y['event']
        )
        
        # Filter features
        pipeline = FeatureSelectionPipeline(filters=[
            FilterCorrelated(h_threshold=0.9),
            FilterByL1(filter_strength=0.03)
        ])

        X_train = pipeline(X_train, y_train)
        feature_columns = X_train.columns.tolist()
        save_feature_columns(feature_columns, model_path.parent / f"features.json")
        
        logger.info("Filtered features: %s", X_train.columns)
        logger.info("Number of features after filtering: %d", X_train.shape[1])
        
    y_structured = Surv.from_dataframe(
        event='event', 
        time='time_to_hit_hours', 
        data=y_train.reset_index(drop=True)
    )
    
    params = read_hyperparameters(hparams_path)  
    model_selection = build_models()
    model = model_selection[model_name](params)
    
    model.fit(X_train, y_structured)

    pd.to_pickle(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    
    


if __name__ == "__main__":
    
    datetime = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
    for model_name in ["rsf", "weibull_aft", "xgb"]:
        logger.info(f"Training {model_name} model...")
        main(
            model_name=model_name, 
            features_path=PROCESSED_DATA_DIR / "train_clean.csv",
            hparams_path=MODELS_DIR / "optim_v1" / f"{model_name}_best_params.json",
            model_path=MODELS_DIR / f"train_v1_{datetime}" / f"{model_name}_model.pkl",
            on_full_train=False,
        )
        
        
