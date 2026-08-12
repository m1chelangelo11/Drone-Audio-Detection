import logging as log
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

logger = log.getLogger(__name__)


def train(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    random_state: int,
    n_estimators: int,
) -> tuple[
    RandomForestClassifier,
    StandardScaler,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Trains a Random Forest model for audio classification and saves
    it and its' scaler in models directory

    Args:
        X: features
        y: target
        groups: an array of group IDs
        random_state: seed for random splitting

    Returns:
        Classifier and scaler objects, y_pred, X_test, y_test,
        X_val, y_val for evaluation
    """

    try:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
        train_idx, temp_idx = next(gss.split(X, y, groups))
        X_train, X_temp = X[train_idx], X[temp_idx]
        y_train, y_temp = y[train_idx], y[temp_idx]

        train_idx, temp_idx = next(gss.split(X_temp, y_temp, groups[temp_idx]))
        X_test, X_val = X_temp[train_idx], X_temp[temp_idx]
        y_test, y_val = y_temp[train_idx], y_temp[temp_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_val = scaler.transform(X_val)

        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight="balanced",
        )
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        root = Path(__file__).resolve().parents[1]

        joblib.dump(classifier, root / "models" / "rf.joblib")
        joblib.dump(scaler, root / "models" / "rf_scaler.joblib")

        return classifier, scaler, y_pred, X_test, y_test, X_val, y_val

    except Exception as e: # noqa: BLE001
        logger.error(f"Error while training: {e}")
        return None
