import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 1 week
TTL = 60 * 60 * 24 * 7


@st.cache_data(ttl=TTL)
def get_data() -> pd.DataFrame:
    """Get example data from kaggle.

    Reference
    ---------
    https://www.kaggle.com/datasets/dineshmanikanta/machine-failure-predictions
    """
    path = "./data/machine_failure.csv"
    data = pd.read_csv(path)

    return data


@st.cache_data(ttl=TTL)
def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(
        columns={
            "Rotational speed [rpm]": "rotational_speed_rpm",
            "Torque [Nm]": "torque_nm",
            "Machine failure": "machine_failure",
        }
    )
    data["temp_ratio"] = data["Process temperature [K]"] / data["Air temperature [K]"]

    columns = ["rotational_speed_rpm", "torque_nm", "temp_ratio", "machine_failure"]

    return data[columns]


@st.cache_resource(ttl=TTL)
def train(data: pd.DataFrame):
    """Train a simple classifier model."""
    y = data["machine_failure"]
    X = data[["rotational_speed_rpm", "torque_nm", "temp_ratio"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )

    clf = RandomForestClassifier(
        max_depth=5, n_estimators=1000, max_samples=0.5, random_state=1
    )
    clf.fit(X_train, y_train)
    # Predict probabilities on the validation set
    y_pred_proba = clf.predict_proba(X_val)[:, 1]

    # Calculate the ROC AUC score
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    st.write(clf)
    st.write(f"ROC AUC score: {roc_auc:.4f}")

    return clf
