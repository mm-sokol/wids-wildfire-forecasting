"""
Code to train models
"""

from logging import getLogger, basicConfig, INFO
from pathlib import Path

from config import MODELS_DIR, PROCESSED_DATA_DIR
from modeling.models import WeibullAFT

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

from features import FilterCorrelated, FilterByL1, FeatureSelectionPipeline

basicConfig(level=INFO)
logger = getLogger(__name__)

def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_clean.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    # Load data
    X = pd.read_csv(features_path)
    y = X[["time_to_hit_hours", "event"]]
    X = X.drop(columns=["time_to_hit_hours", "event"])
    
    
    # Filter features
    pipeline = FeatureSelectionPipeline(filters=[
        FilterCorrelated(h_threshold=0.9),
        # FilterTargetCorrelated(l_threshold=0.1),
        FilterByL1(filter_strength=0.03)
    ])

    X_train = pipeline(X, y)
    logger.info("Filtered features: %s", X_train.columns)
    logger.info("Number of features after filtering: %d", X_train.shape[1])
    
    np.random.seed(101)
    kf = KFold(n_splits=5, shuffle=True, random_state=101)
    all_predictions = []

    logger.info("Starting training loop across 5 folds...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, _ = y.iloc[train_idx], y.iloc[val_idx]


        model = WeibullAFT()
        model.fit(X_train_fold, y_train_fold)

        # Concordance Index (C-index)
        # 0.5 is random guessing
        # 1.0 is perfect prediction
        c_index = model._model.concordance_index_
        logger.info(f"Fold {fold + 1} - Model Training C-index: {c_index:.4f}")

        val_preds = model.predict(X_val_fold)
        all_predictions.append(val_preds)

    oof_predictions = pd.concat(all_predictions).sort_index()
    logger.info("\nTraining done.")
    logger.info(oof_predictions.head())
    


if __name__ == "__main__":
    main()
