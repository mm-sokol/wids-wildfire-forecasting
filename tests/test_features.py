
import pytest
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from src.config import RAW_DATA_DIR
from src.features import (
    filter_target_uncorrelated_features,
    filter_correlated_features,
    filter_features_by_L1_regularization,
    filter_features_by_RFECV,
    # reduce_features_by_PCA,
)

@pytest.fixture
def sample_training_data():
    
    df = pd.read_csv(RAW_DATA_DIR / "train.csv")

    unused = ["event", "time_to_hit_hours", "event_id"]

    X_train, _, y_train, _ = train_test_split(
        df[[c for c in df.columns if c not in unused]], 
        df['event'], 
        test_size=0.2, 
        random_state=222,
        stratify=df['event']
    )
    
    return X_train, y_train

@pytest.fixture
def sample_model():
    
   return RandomForestClassifier(n_estimators=400, max_depth=2)
    


def test_filter_target_uncorrelated(sample_training_data):
    X, y = sample_training_data

    X_filtered = filter_target_uncorrelated_features(X, y, l_threshold=0.1)

    assert isinstance(X_filtered, pd.DataFrame)
    assert X_filtered.shape[1] < X.shape[1]
    assert not X_filtered.empty
    
def test_filter_correlated_features(sample_training_data):
    X, y = sample_training_data
    
    X_filtered = filter_correlated_features(X, h_threshold=0.85)
    
    assert isinstance(X_filtered, pd.DataFrame)
    assert X_filtered.shape[1] < X.shape[1]
    assert not X_filtered.empty
    
def test_filter_features_by_L1_regularization(sample_training_data):
    X, y = sample_training_data
    
    X_filtered = filter_features_by_L1_regularization(X, y, filter_strength=0.1)
    
    assert isinstance(X_filtered, pd.DataFrame)
    assert X_filtered.shape[1] < X.shape[1]
    assert not X_filtered.empty
    
def test_filter_features_by_RFECV(sample_model, sample_training_data):
    
    X, y = sample_training_data
    
    X_filtered = filter_features_by_RFECV(sample_model, X, y)
    
    assert isinstance(X_filtered, pd.DataFrame)
    assert X_filtered.shape[1] < X.shape[1]
    assert not X_filtered.empty