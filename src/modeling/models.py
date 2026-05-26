import numpy as np
import pandas as pd
from pathlib import Path

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import brier_score
from lifelines import WeibullAFTFitter
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

HORIZONS = [12, 24, 48, 72]

def _predict_sksurv(model, X, cols):
    surv_funcs = model.predict_survival_function(X[cols])
    rows = []

    for sf in surv_funcs:
        rows.append({
            f"prob_{t}h": 1.0 - float(sf(min(t, sf.x[-1])))
            for t in HORIZONS
        })

    return pd.DataFrame(rows)

class RSFModel:
    
    def __init__(self, n_estimators=300, min_samples_split=10, min_samples_leaf=6, max_features="sqrt"):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
    
    def fit(self, X, y):
        self._cols = list(X.columns)
        self._model = RandomSurvivalForest(
            n_estimators=self.n_estimators, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features, n_jobs=-1, random_state=42,
        )
        self._model.fit(X[self._cols], y)
        return self

    def predict(self, X):
        return _predict_sksurv(self._model, X, self._cols)

    def feature_importances(self):
        return pd.Series(self._model.feature_importances_, index=self._cols).sort_values(ascending=False)

class WeibullAFT:
    def __init__(self, penalizer=0.1):
        self.penalizer = penalizer

    def fit(self, X, y):
        self._cols = list(X.columns)
        df = X.copy()
        df["time_to_hit_hours"] = y["time_to_hit_hours"]
        df["event"] = y["event"].astype(int)
        self._model = WeibullAFTFitter(penalizer=self.penalizer)

        self._model.fit(df, duration_col="time_to_hit_hours", event_col="event")
        return self

    def predict(self, X):
        sf = self._model.predict_survival_function(X[self._cols])
        rows = {}
        for t in HORIZONS:
            idx = min(np.searchsorted(sf.index.values, t), len(sf) - 1)
            rows[f"prob_{t}h"] = 1.0 - sf.iloc[idx].values
        return pd.DataFrame(rows)

class XGBSurvival:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

    def fit(self, X, y):
        self._cols = list(X.columns)
        labels = np.where(y["event"], y["time_to_hit_hours"], -y["time_to_hit_hours"])
        self._model = xgb.XGBRegressor(
            objective="survival:cox",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42
        )
        self._model.fit(X[self._cols], labels)
        return self

    def predict(self, X):
        risk_scores = self._model.predict(X[self._cols])
        rows = {f"prob_{t}h": [] for t in HORIZONS}
        for score in risk_scores:
            for t in HORIZONS:
                p = 1.0 - np.exp(-np.exp(score) * (t / 72.0))
                rows[f"prob_{t}h"].append(float(np.clip(p, 0.01, 0.99)))
        return pd.DataFrame(rows)

class Ensemble:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [m.predict(X) for m in self.models.values()]
        return sum(preds) / len(preds)

def build_models():
    return {
        "rsf": RSFModel(),
        "weibull_aft": WeibullAFT(),
        "xgb": XGBSurvival(),
    }

def fit_all(models, X, y):
    for name, model in models.items():
        model.fit(X, y)

    return models

def predict_all(models, X):
    return {name: model.predict(X) for name, model in models.items()}

def ensemble_preds(preds):
    return sum(preds.values()) / len(preds)

def cross_validate_brier(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    records = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y["event"])):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        models = build_models()
        fit_all(models, X_tr, y_tr)
        preds = predict_all(models, X_val)
        preds["ensemble"] = ensemble_preds(preds)

        for name, pred_df in preds.items():
            for t in HORIZONS:
                surv = 1.0 - pred_df[f"prob_{t}h"].clip(1e-6, 1 - 1e-6).values
                try:
                    bs = brier_score(y_tr, y_val, surv.reshape(-1, 1), [float(t)])[1][0]
                except Exception:
                    bs = np.nan
                records.append({"fold": fold, "model": name, "horizon": t, "brier_score": bs})

    return (
        pd.DataFrame(records)
        .groupby(["model", "horizon"])["brier_score"]
        .mean()
        .unstack("horizon")
        .round(4)
    )