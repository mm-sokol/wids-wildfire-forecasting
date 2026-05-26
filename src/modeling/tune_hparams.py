from pathlib import Path

from sklearn.model_selection import KFold, StratifiedKFold
from config import MODELS_DIR, PROCESSED_DATA_DIR

from optuna import Trial, create_study
from optuna.samplers import TPESampler, CmaEsSampler
import json
from models import RSFModel, WeibullAFT, XGBSurvival
import pandas as pd
from sksurv.util import Surv
from features import FilterCorrelated, FilterByL1, FeatureSelectionPipeline, FilterTargetCorrelated
from sksurv.metrics import concordance_index_censored
from lifelines.utils import concordance_index

def objective(trial: Trial, model_name: str, X, y, n_splits: int = 5):
    
    if model_name == "rsf":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        model = RSFModel(n_estimators=n_estimators, min_samples_split=min_samples_split,
                         min_samples_leaf=min_samples_leaf, max_features=max_features)
    elif model_name == "weibull_aft":
        penalizer = trial.suggest_float("penalizer", 0.01, 1.0, log=True)
        model = WeibullAFT(penalizer=penalizer)
    elif model_name == "xgb":
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        model = XGBSurvival(n_estimators=n_estimators, learning_rate=learning_rate,
                            max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y["event"])):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        model.fit(X_tr, y_tr)
        _ = model.predict(X_val)
        if model_name == "rsf":
            c_index = model._model.score(X_val, y[val_idx])
        elif model_name == "weibull_aft":
            predicted_times = model.predict_median(X_val)

            c_index = concordance_index(
                actual_time=y_val['time_to_hit_hours'], 
                predicted_time=predicted_times, 
                event_observed=y_val['event'] 
            )
        elif model_name == "xgb":
            risk_scores = model._model.predict(X_val[model._cols])
            c_index = concordance_index_censored(y_val['event'], y_val['time_to_hit_hours'], -risk_scores)[0]
            
        records.append(c_index)

    mean_c_index = sum(records) / len(records)
    return mean_c_index


def tune_hyperparameters(model_name: str, X, y, n_trials: int = 50, save_path: Path = MODELS_DIR):

    study = create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, model_name, X, y), n_trials=n_trials)

    print(f"Best hyperparameters for {model_name}: {study.best_params}")
    print(f"Best C-index for {model_name}: {study.best_value}")
    
    saved_path = save_path / f"{model_name}_best_params.json"
    with open(saved_path, "w") as f:
        json.dump(study.best_params, f)
    print(f"Saved best hyperparameters for {model_name} to {saved_path}")



def main(
    features_path: Path = PROCESSED_DATA_DIR / "train_clean.csv",
    model_path: Path = MODELS_DIR,
):
    X = pd.read_csv(features_path)
    y = X[["event", "time_to_hit_hours"]]
    X = X.drop(columns=["time_to_hit_hours", "event"])

    y_structured = Surv.from_dataframe(
        event='event', 
        time='time_to_hit_hours', 
        data=y
    )
    
    # Apply feature selection
    pipeline = FeatureSelectionPipeline(filters=[
        FilterCorrelated(h_threshold=0.9),
        # FilterTargetCorrelated(l_threshold=0.001),
        FilterByL1(filter_strength=0.03)
    ])
    
    X = pipeline(X, y)
    print(f"Number of features after selection: {X.shape[1]}")

    # return
    for model_name in ["rsf", "weibull_aft", "xgb"]:
        print(f"Tuning hyperparameters for {model_name}...")
        
        tune_hyperparameters(model_name, X, y_structured, n_trials=50, save_path=model_path)
        
        


if __name__ == "__main__":
    main()