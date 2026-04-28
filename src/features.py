"""
Code to create features for modeling
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import check_cv, train_test_split
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA

from abc import ABC, abstractmethod
from logging import getLogger

from config import PROCESSED_DATA_DIR

logger = getLogger(__name__)

class FeatureFilterBase(ABC):
    
    @abstractmethod
    def apply(X_train: pd.DataFrame) -> pd.DataFrame:
        pass
    
    
class FilterCorrelated(FeatureFilterBase):
    
    def __init__(self, h_threshold: float):
        super().__init__()
        self.h_threshold = h_threshold
    
    def apply(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        
        corr = X_train.corr(method='spearman').abs()
        corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in corr_upper.columns if any(corr_upper[c] > self.h_threshold)]
        
        return X_train.drop(columns=to_drop), y_train


class FilterTargetCorrelated(FeatureFilterBase):
    
    def __init__(self, l_threshold: float):
        super().__init__()
        self.l_threshold = l_threshold

    def apply(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        
        corr = X_train.corrwith(y_train, method='spearman').abs()
        selected_columns = corr[corr > self.l_threshold].index
        return X_train[selected_columns], y_train


class FilterByL1(FeatureFilterBase):
    
    def __init__(self, filter_strength: float):
        super().__init__()
        self.filter_strength = filter_strength

    def apply(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        
        lasso = Lasso(alpha=self.filter_strength)
        selector = SelectFromModel(lasso)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()]
        return X_train[selected_features], y_train


class ReduceByPCA(FeatureFilterBase):

    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        
    def apply(self, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        
        n_features_actual = self.n_features
        n_samples, n_data_features = X_train.shape
        limit = min(n_samples, n_data_features)
            
        if self.n_features > limit:
            logger.warning("Given n_features (%d) exceeds the theoretical limit (%d)", self.n_features, limit)
            logger.warning("PCA will reduce dimensionality to the limit number of dimensions.")
            n_features_actual = limit

        elif n_data_features < 2:
            logger.warning("Number of X_train features ({}) is too low. PCA will not reduce dimensionality.", n_data_features)
            n_features_actual = n_data_features
            
        elif self.n_features < n_data_features:
            logger.info("Number of X_train features ({}) is not greater than n_features ({}) given to PCA", n_data_features, self.n_features)
            logger.info("PCA will reduce dimensionality by one.")
            n_features_actual = self.n_features-1
            
            
        pca = PCA(n_components=n_features_actual)
        X_pca = pca.fit_transform(X_train)
        columns = [f"PCA_{i+1}" for i in range(n_features_actual)]
        return pd.DataFrame(X_pca, columns=columns, index=X_train.index), y_train


class FilterByRFECV():

    def __init__(self,
        model: BaseEstimator,
        selection_metric: str = "accuracy",
        k_folds: int = 5,
        min_features: int = 5,
        step: int = 1,
        is_classification_task: bool = False
    ):
        super().__init__()
        self.model = model
        self.selection_metric = selection_metric
        self.k_folds = k_folds
        self.min_features = min_features
        self.step = step
        self.classifer = is_classification_task

    def apply(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
     
        cv = check_cv(self.k_folds, y_train, classifier=self.classifier) 
        selector = RFECV(
            estimator=self.model,
            step=self.step,
            cv=cv,
            scoring=self.selection_metric,
            min_features_to_select=self.min_features,
            n_jobs=-1
        )
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.support_]

        return X_train[selected_features], y_train

class FeatureSelectionPipeline():
    
    def __init__(self, filters: list[FeatureFilterBase]):
        self.filters = filters
    
    def __call__(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
        if len(y_train) != len(X_train):
            logger.error("Target series and train data have length mismatch ({}!={})", len(y_train), len(X_train))
            
        for f in self.filters:
            X_train, y_train = f.apply(X_train, y_train)
        return X_train



if __name__ == "__main__":


    df = pd.read_csv(PROCESSED_DATA_DIR / "train_clean.csv")

    unused = ["event", "time_to_hit_hours", "event_id"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[[c for c in df.columns if c not in unused]], 
        df['time_to_hit_hours'], 
        test_size=0.2, 
        random_state=222,
    )

    pipeline = FeatureSelectionPipeline(filters=[
        FilterCorrelated(h_threshold=0.9),
        FilterTargetCorrelated(l_threshold=0.1),
        ReduceByPCA(n_features=12)
    ])
    
    X_train_filtered = pipeline(X_train, y_train)
    print(X_train_filtered.columns)
