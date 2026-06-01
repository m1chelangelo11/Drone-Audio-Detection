from sklearn.metrics import (accuracy_score,
                             confusion_matrix,
                             roc_auc_score,
                             average_precision_score,
                             classification_report)
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple
import numpy as np
import logging as log

def evaluation(X_test: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray,
               classifier: RandomForestClassifier, X_val: np.ndarray,
               y_val: np.ndarray) -> Tuple:
    """Evaluates model using accuracy score, confusion matrix, roc_auc
    score, pr_auc score, precission, recall and f1 score for each test 
    and validation predictions

    Args:
        X_test: features test subset
        y_test: target test subset
        classifier: saved Random Forest model
        X_val: features validation subset
        y_val: target validation subset
    
    Returns:
        Tuple of all the results
    """

    try:
        predict_proba = classifier.predict_proba(X_test)[:, 1]
        predict_proba_val = classifier.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, predict_proba)
        pr_auc = average_precision_score(y_test, y_pred)
        cr = classification_report(y_test, y_pred)

        y_val_pred = classifier.predict(X_val)
        acc_val = accuracy_score(y_val, y_val_pred)
        conf_matrix_val = confusion_matrix(y_val, y_val_pred)
        roc_auc_val = roc_auc_score(y_val, predict_proba_val)
        pr_auc_val = average_precision_score(y_val, y_val_pred)
        cr_val = classification_report(y_val, y_val_pred)

        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test Confusion Matrix: {conf_matrix}")
        print(f"Test ROC_AUC: {roc_auc:.4f}")
        print(f"Test PR_AUC: {pr_auc:.4f}")
        print(f"Test Classification Report: {cr}")

        print(f"Validation Accuracy: {acc_val:.4f}")
        print(f"Validation Confusion Matrix: {conf_matrix_val}")
        print(f"Validation ROC_AUC: {roc_auc_val:.4f}")
        print(f"Validation PR_AUC: {pr_auc_val:.4f}")
        print(f"Validation Classification Report: {cr_val}")

        return (acc, acc_val, conf_matrix, conf_matrix_val,
                roc_auc,roc_auc_val, pr_auc, pr_auc_val, cr, cr_val)
    
    except Exception as e:
        log.error(f"Error while evaluating: {e}")
        return None