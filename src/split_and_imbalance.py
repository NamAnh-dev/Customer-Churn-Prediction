"""Splitting and imbalance-handling utilities.

Two things fixed here relative to the original `preprocessor.py`:

1. `split_data` now supports a train/val/test split. The validation set is
   what threshold tuning should run against — the test set is only touched
   once, at the very end, to report final numbers. Tuning the threshold on
   the test set and then reporting metrics on that same test set (as the
   original notebook did) overstates how well the model will generalize.

2. `compare_imbalance_strategies` now runs SMOTE/under-sampling on an
   already fully-numeric, already-transformed feature matrix (the output of
   `preprocessing.build_preprocessor()`), so there is no "reattach the
   non-numeric columns by row position" step — that step in the original
   code silently misaligned rows because SMOTE's synthetic rows don't
   correspond 1:1 with the original row order.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train/val/test split.

    val_size and test_size are both fractions of the *original* dataset.
    Use X_val/y_val for threshold tuning and model selection; touch
    X_test/y_test only for the final reported metrics.
    """
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


def compare_imbalance_strategies(
    X_train: np.ndarray, y_train: pd.Series, random_state: int = 42
) -> dict[str, tuple[np.ndarray, pd.Series]]:
    """Return {strategy_name: (X, y)} for a fully-numeric, already-encoded X_train.

    This is exploratory: use it to justify *why* you picked a strategy
    (e.g. "SMOTE and class_weight=balanced gave near-identical recall, so we
    kept class_weight for simplicity — no synthetic data to maintain").
    It is not meant to feed the final production pipeline directly.
    """
    strategies = {"Original (imbalanced)": (X_train, y_train)}

    try:
        from imblearn.over_sampling import SMOTE
        X_sm, y_sm = SMOTE(random_state=random_state).fit_resample(X_train, y_train)
        strategies["SMOTE"] = (X_sm, y_sm)
    except ImportError:
        log.warning("imbalanced-learn not installed (pip install imbalanced-learn); skipping SMOTE")

    try:
        from imblearn.under_sampling import RandomUnderSampler
        X_rus, y_rus = RandomUnderSampler(random_state=random_state).fit_resample(X_train, y_train)
        strategies["Under-sampling"] = (X_rus, y_rus)
    except ImportError:
        pass

    return strategies
