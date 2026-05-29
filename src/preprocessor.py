import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

log = logging.getLogger(__name__)


# ── Encoding ───────────────────────────────────────────────────────────────
def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in service_cols:
        if col in X.columns:
            X[col] = X[col].replace("No internet service", "No")

    if "MultipleLines" in X.columns:
        X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")

    binary_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "PaperlessBilling",
    ] + service_cols

    binary_map = {"Yes": 1, "No": 0, "Female": 0, "Male": 1}
    for col in binary_cols:
        if col in X.columns:
            X[col] = X[col].map(binary_map).fillna(0).astype(int)

    if "Contract" in X.columns:
        contract_order = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        X["Contract"] = X["Contract"].map(contract_order).fillna(0).astype(int)

    ohe_cols = []
    if "InternetService" in X.columns:
        ohe_cols.append("InternetService")
    if "PaymentMethod" in X.columns:
        ohe_cols.append("PaymentMethod")

    if ohe_cols:
        X = pd.get_dummies(X, columns=ohe_cols, drop_first=False, dtype=int)

    log.info(f"Encoded features: {X.shape[1]} columns")
    return X


# ── Feature Engineering ───────────────────────────────────────────────────
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
 
    X = X.copy()

    if "TotalCharges" in X.columns and "tenure" in X.columns:
        X["avg_monthly_charge"] = np.where(
            X["tenure"] > 0,
            X["TotalCharges"] / X["tenure"],
            X.get("MonthlyCharges", 0),
        )

    if "MonthlyCharges" in X.columns and "TotalCharges" in X.columns:
        X["charge_ratio"] = np.where(
            X["TotalCharges"] > 0,
            X["MonthlyCharges"] / X["TotalCharges"],
            1.0,
        )

    service_binary_cols = [
        c for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies",
                    "PhoneService", "MultipleLines"]
        if c in X.columns and X[c].dtype in [int, float]
    ]
    if service_binary_cols:
        X["num_services"] = X[service_binary_cols].sum(axis=1)

    if "tenure" in X.columns:
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
            include_lowest=True,
        ).astype(str)

    log.info(f"After feature engineering: {X.shape[1]} columns")
    return X


# ── Train/Test Split ──────────────────────────────────────────────────────
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,  # QUAN TRỌNG với imbalanced data
    )
    log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
    log.info(f"Train churn rate: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")
    return X_train, X_test, y_train, y_test


# ── Imbalanced data strategies ────────────────────────────────────────────
def get_imbalanced_strategies(X_train: pd.DataFrame, y_train: pd.Series) -> dict:
    strategies = {}

    # ── Strategy 1: Original (baseline) ──────────────────────────────────
    strategies["Original (imbalanced)"] = (X_train.copy(), y_train.copy())

    # ── Strategy 2: SMOTE ─────────────────────────────────────────────────
    try:
        from imblearn.over_sampling import SMOTE
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        X_num = X_train[numeric_cols]

        sm = SMOTE(random_state=42, k_neighbors=5)
        X_sm, y_sm = sm.fit_resample(X_num, y_train)

        # Rebuild full DataFrame với các cột đã drop
        X_sm_full = pd.DataFrame(X_sm, columns=numeric_cols)
        for col in X_train.columns:
            if col not in numeric_cols:
                X_sm_full[col] = X_train[col].values[
                    :len(X_sm_full) if len(X_sm_full) <= len(X_train) else len(X_train)
                ]

        strategies["SMOTE"] = (X_sm_full[X_train.columns], y_sm)
        log.info(f"SMOTE: {y_sm.value_counts().to_dict()}")

    except ImportError:
        log.warning("imbalanced-learn not installed. Run: pip install imbalanced-learn")
        log.warning("Skipping SMOTE strategy.")

    # ── Strategy 3: Random Under-sampling ────────────────────────────────
    try:
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        strategies["Under-sampling"] = (X_rus, y_rus)
        log.info(f"Under-sampling: {y_rus.value_counts().to_dict()}")
    except ImportError:
        pass

    return strategies
