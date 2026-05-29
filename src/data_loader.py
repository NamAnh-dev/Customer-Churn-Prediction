import pandas as pd
import logging

log = logging.getLogger(__name__)


def load_telco(path: str = "data/telco_churn.csv") -> pd.DataFrame:

    df = pd.read_csv(path)
    log.info(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].astype(float)
    
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    log.info(f"Churn rate: {df['Churn'].mean():.1%} "
             f"({df['Churn'].sum():,} churned / {len(df):,} total)")

    return df

def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


