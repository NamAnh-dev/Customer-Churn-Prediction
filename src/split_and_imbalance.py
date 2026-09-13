from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

def split_data(X: pd.DataFrame, y: pd.Series, val_size: float = 0.2, test_size: float = 0.2, random_state: int = 42):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, random_state=random_state, stratify=y
    )
    relative_test_size = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test_size, random_state=random_state, stratify=y_temp
    )
    log.info(
        "Train: %s | Val: %s | Test: %s", f"{len(X_train):,}", f"{len(X_val):,}", f"{len(X_test):,}"
    )
    log.info(
        "Churn rate — train: %.1f%% | val: %.1f%% | test: %.1f%%",
        100 * y_train.mean(), 100 * y_val.mean(), 100 * y_test.mean(),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def compare_imbalance_strategies(X_train: np.ndarray, y_train: pd.Series, random_state: int = 42) -> dict[str, tuple[np.ndarray, pd.Series]]:
    strategies = {"Original (imbalanced)": (X_train, y_train)}

    try:
        from imblearn.over_sampling import SMOTE
        X_sm, y_sm = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        strategies["SMOTE"] = (X_sm, y_sm)
    except ImportError:
        pass

    try:
        from imblearn.under_sampling import RandomUnderSampler
        X_rus, y_rus = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
        strategies["Under-sampling"] = (X_rus, y_rus)
    except ImportError:
        pass

    return strategies
