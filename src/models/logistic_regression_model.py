"""
Logistic Regression Classifier
Implementation of Logistic Regression for IoT botnet detection
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class LogisticRegressionModel:
    """Logistic Regression classifier for botnet detection"""

    def __init__(self, **params):
        """
        Initialize Logistic Regression model

        Args:
            **params: Parameters for LogisticRegression
        """
        defaults = {'max_iter': 1000, 'random_state': 42, 'solver': 'lbfgs'}
        defaults.update(params)
        self.params = defaults
        self.model = LogisticRegression(**self.params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, float]:
        """Train the Logistic Regression model"""
        logger.info("Training Logistic Regression model...")
        logger.info(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

        self.model.fit(X_train, y_train)

        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train).mean()
        metrics = {'train_accuracy': train_accuracy}

        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = (val_pred == y_val).mean()
            metrics['val_accuracy'] = val_accuracy
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")

        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info("Training completed!")
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """Evaluate model on test data"""
        logger.info("Evaluating Logistic Regression model...")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        accuracy = (y_pred == y_test).mean()
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        try:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except Exception:
            roc_auc = None

        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'probabilities': y_proba
        }

        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        if roc_auc:
            logger.info(f"ROC-AUC: {roc_auc:.4f}")

        if return_predictions:
            metrics['predictions'] = y_pred

        return metrics

    def save_model(self, save_path: str):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, load_path: str):
        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")

    def __repr__(self):
        return f"LogisticRegressionModel(params={self.params})"
