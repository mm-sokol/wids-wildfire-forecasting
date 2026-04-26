"""
Code to create features for modeling
"""

from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.model_selection import check_cv
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA



def filter_correlated_features(
    X_train: pd.DataFrame,
    h_threshold: float
) -> pd.DataFrame:
    """Function gets rid of features that are highly correlated with each other,
    based on absolute value of Spearman's correlation.

    Args:
        X_train (pd.DataFrame): training dataframe
        y_train (pd.Series): training labels
        h_threshold (float): the upper threshold for correlation between features

    Returns:
        pd.DataFrame: the modified training dataframe, without correlated feeatures
    """

    corr = X_train.corr(method='spearman').abs()
    corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in corr_upper.columns if any(corr_upper[c] > h_threshold)]
    
    return X_train.drop(columns=to_drop)


def filter_target_uncorrelated_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    l_threshold: float
) -> pd.DataFrame:
    """Function gets rid of columns that have Spearman's correlation value
    with our target (y_train) below given theshold.

    Args:
        X_train (pd.DataFrame): training dataframe
        y_train (pd.Series): training labels
        h_threshold (float): the lower threshold for correlation to target
        
    Returns:
        pd.DataFrame: the modified training dataframe, without correlated feeatures
    """
    
    corr = X_train.corrwith(y_train, method='spearman').abs()
    selected_columns = corr[corr > l_threshold].index
    return X_train[selected_columns]


def filter_features_by_L1_regularization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    filter_strength: float
) -> pd.DataFrame:
    """Function gets rid of columns that have Spearman's correlation value
    with our target (y_train) below given theshold.

    Args:
        X_train (pd.DataFrame): training dataframe
        y_train (pd.Series): training labels
        h_threshold (float): the lower threshold for correlation to target
        
    Returns:
        pd.DataFrame: the modified training dataframe, without correlated feeatures
    """
     
    lasso = Lasso(alpha=filter_strength)
    selector = SelectFromModel(lasso)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    return X_train[selected_features]


def reduce_features_by_PCA(
    X_train: pd.DataFrame,
    n_features: float
) -> pd.DataFrame:
    """Function reduces number of features to a given number,
    by selecting the directions, that explain the most variance in data (PCA).

    Args:
        X_train (pd.DataFrame): training dataframe
        n_features (float): wanted number of features

    Returns:
        pd.DataFrame: training dataframe with PCA component features
    """

    pca = PCA(n_components=n_features)
    X_pca = pca.fit_transform(X_train)

    columns = [f"PCA_{i+1}" for i in range(n_features)]
    return pd.DataFrame(X_pca, columns=columns, index=X_train.index)


def filter_features_by_RFECV(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selection_metric: str = "accuracy",
    k_folds: int = 5,
    min_features: int = 5,
    step: int = 1
):
    """Function filters features by checking their importance for 
    given model by recursive feature elimination using k-fold crossvalidation.

    Args:
        model (BaseEstimator): the model we'd like to use for classificaiton
        X_train (pd.DataFrame): training dataframe
        y_train (pd.Series): training labels
        selection_metric (str, optional): the metric, we want to maximize in crossvalidation. Defaults to "accuracy".
        k_folds (int, optional): Number of crossvalidaiton folds. Defaults to 5.
        min_features (int, optional): the minimum number of features. Defaults to 5.
        step (int, optional): the number of features which the model eliminates with each training run. Defaults to 1.

    Returns:
        _type_: _description_
    """

    cv = check_cv(k_folds, y_train, classifier=True) 
    selector = RFECV(
        estimator=model,
        step=step,
        cv=cv,
        scoring=selection_metric,
        min_features_to_select=min_features,
        n_jobs=-1
    )
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]

    return X_train[selected_features]

if __name__ == "__main__":
    pass
