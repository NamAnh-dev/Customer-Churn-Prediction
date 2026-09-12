from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

log = logging.getLogger(__name__)

SERVICE_COLS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "MultipleLines"] + SERVICE_COLS
ORDINAL_CONTRACT_ORDER = ["Month-to-month", "One year", "Two year"]
ONEHOT_COLS = ["InternetService", "PaymentMethod", "tenure_group"]
NUMERIC_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "avg_monthly_charge", "charge_ratio", "num_services",
]


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None) -> "ChurnFeatureEngineer":
        return self  # stateless: nothing to learn from the training data

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col in SERVICE_COLS:    
            if col in X.columns:
                X[col] = X[col].replace("No internet service", "No")
        if "MultipleLines" in X.columns:
            X["MultipleLines"] = X["MultipleLines"].replace("No phone service", "No")

        if {"TotalCharges", "tenure"}.issubset(X.columns):
            X["avg_monthly_charge"] = np.where(
                X["tenure"] > 0, X["TotalCharges"] / X["tenure"], X.get("MonthlyCharges", 0)
            )
        if {"MonthlyCharges", "TotalCharges"}.issubset(X.columns):
            X["charge_ratio"] = np.where(
                X["TotalCharges"] > 0, X["MonthlyCharges"] / X["TotalCharges"], 1.0
            )

        service_flag_cols = [c for c in SERVICE_COLS + ["PhoneService", "MultipleLines"] if c in X.columns]
        if service_flag_cols:
            X["num_services"] = X[service_flag_cols].eq("Yes").sum(axis=1)

        if "tenure" in X.columns:
            X["tenure_group"] = pd.cut(
                X["tenure"],
                bins=[0, 12, 24, 48, 72],
                labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
                include_lowest=True,
            ).astype(str)

        return X


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("gender", OrdinalEncoder(categories=[["Female", "Male"]]), ["gender"]),
            (
                "yes_no",
                OrdinalEncoder(categories=[["No", "Yes"]] * len(YES_NO_COLS),
                                handle_unknown="use_encoded_value", unknown_value=-1),
                YES_NO_COLS,
            ),
            (
                "contract",
                OrdinalEncoder(categories=[ORDINAL_CONTRACT_ORDER],
                                handle_unknown="use_encoded_value", unknown_value=-1),
                ["Contract"],
            ),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ONEHOT_COLS),
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("passthrough_binary", "passthrough", ["SeniorCitizen"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_full_pipeline(model) -> Pipeline:
    return Pipeline([
        ("features", ChurnFeatureEngineer()),
        ("preprocess", build_preprocessor()),
        ("model", model),
    ])
