
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import brier_score

from models.random_survival_forest import RSFModel
from models.weibull_aft import WeibullAFT
from models.xgb_survival import XGBSurvival

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


def build_models():
    return {
        "rsf": lambda params: RSFModel(**params),
        "weibull_aft": lambda params: WeibullAFT(**params),
        "xgb": lambda params: XGBSurvival(**params),
    }

def fit_all(models, X, y):
    for model in models.values():
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
                except RuntimeError:
                    bs = np.nan
                records.append({"fold": fold, "model": name, "horizon": t, "brier_score": bs})

    return (
        pd.DataFrame(records)
        .groupby(["model", "horizon"])["brier_score"]
        .mean()
        .unstack("horizon")
        .round(4)
    )