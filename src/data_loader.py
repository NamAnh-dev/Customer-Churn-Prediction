from pathlib import Path
import pandas as pd
import logging

log = logging.getLogger(__name__)


def load_telco(path: str | Path = "../data/telco_churn.csv"):
    df = pd.read_csv(path)
    log.info("Loaded %s rows x %s cols from %s", f"{df.shape[0]:,}", df.shape[1], path)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    n_missing = df["TotalCharges"].isna().sum()
    if n_missing:
        log.info("Filling %d missing TotalCharges (tenure=0 customers) with 0", n_missing)
        df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    log.info(
        "Churn rate: %.1f%% (%s churned / %s total)",
        100 * df["Churn"].mean(),
        f"{df['Churn'].sum():,}",
        f"{len(df):,}",
    )
    return df


def split_features_target(df: pd.DataFrame, target: str = "Churn") -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target])
    y = df[target]
    return X, y
