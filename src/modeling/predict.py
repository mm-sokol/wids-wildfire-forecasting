"""
Code to run model inference with trained models
"""

import json
from pathlib import Path
import pickle

from config import MODELS_DIR, PROCESSED_DATA_DIR, seed_everything, SEED
from features import FeatureSelectionPipeline, FilterByL1, FilterCorrelated
from modeling.utils import HORIZONS
import numpy as np
import pandas as pd
from logging import getLogger, basicConfig, INFO
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, integrated_brier_score, brier_score
from sksurv.util import Surv
from sklearn.model_selection import train_test_split

basicConfig(level=INFO)
logger = getLogger(__name__)

def read_model(model_path):
    with open(model_path, 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def test(
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
        y_train_structured = Surv.from_dataframe(
            event='event', 
            time='time_to_hit_hours', 
            data=y_train.reset_index(drop=True)
        )
        y_test_structured = Surv.from_dataframe(
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

    time_max = min(y_train["time_to_hit_hours"].max(), y_test["time_to_hit_hours"].max()) - 1e-5
    time_min = max(y_train["time_to_hit_hours"].min(), y_test["time_to_hit_hours"].min())
    
    risk_scores = model._model.predict(X_test)
    print(risk_scores)
    c_index_uno, _, _, _, _ = concordance_index_ipcw(
        y_train_structured, y_test_structured, risk_scores, tau=time_max
    )

    times = np.linspace(time_min, time_max, num=100)
    # horizons_array = np.array(HORIZONS)
        
    surv_funcs = model._model.predict_survival_function(X_test)
    estimates = np.vstack([fn(times) for fn in surv_funcs])

    times_out, brier_scores = brier_score(y_train_structured, y_test_structured, estimates, times)
    ibs = integrated_brier_score(y_train_structured, y_test_structured, estimates, times)
    
    model_metrics = {
        "model": model_path,
        "unos_c_index": c_index_uno,
        "ibs": ibs
    }

    for horizon, score in zip(HORIZONS, brier_scores):
        
        model_metrics[f"{horizon}h_barrier_score"] = score
        print(f"Brier Score at horizon {horizon}: {score:.4f}")
        
    return model_metrics
        
    


if __name__ == "__main__":
    
    models_path = MODELS_DIR / "train_v1_2026-06-18_08-47-17"
    pkl_files = [f for f in models_path.iterdir() if f.suffix == ".pkl"]
    
    model_comparison = []
    for pkl in pkl_files:

        metrics = test(
            features_path = PROCESSED_DATA_DIR / "train_clean.csv",
            feature_subset = models_path / "features.json",
            model_path = pkl,
            on_full_test = False,
        )
        print(metrics)
        model_comparison.append(metrics)
        
    model_comparison_df = pd.DataFrame(model_comparison)
    model_comparison_df.to_csv(models_path / "models_evaluation.csv")
