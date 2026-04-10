"""
Support Vector Machine (SVM) Classifier
Implementation of SVM for IoT botnet detection
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class SVMModel:
    """SVM classifier for botnet detection"""
    
    def __init__(self, **params):
        """
        Initialize SVM model
        
        Args:
            **params: Parameters for SVC
        """
        # Set probability=True for probability estimates
        if 'probability' not in params:
            params['probability'] = True
            
        self.model = SVC(**params)
        self.params = params
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Train the SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training SVM model...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_train)
        train_accuracy = (train_pred == y_train).mean()
        
        metrics = {
            'train_accuracy': train_accuracy
        }
        
        # Validation metrics if provided
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = (val_pred == y_val).mean()
            metrics['val_accuracy'] = val_accuracy
            
            logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        logger.info(f"Training accuracy: {train_accuracy:.4f}")
        logger.info(f"Number of support vectors: {len(self.model.support_)}")
        logger.info("Training completed!")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted labels
        """
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        if not self.params.get('probability', False):
            logger.warning("Model was not trained with probability=True")
            return None
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        return_predictions: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating SVM model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC-AUC (if probability estimates available)
        roc_auc = None
        if self.params.get('probability', False):
            try:
                y_proba = self.predict_proba(X_test)
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            except:
                pass
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report,
            'n_support_vectors': len(self.model.support_)
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        if roc_auc:
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        if return_predictions:
            metrics['predictions'] = y_pred
            if self.params.get('probability', False):
                metrics['probabilities'] = self.predict_proba(X_test)
        
        return metrics
    
    def get_support_vectors(self):
        """
        Get support vectors
        
        Returns:
            Support vectors
        """
        return self.model.support_vectors_
    
    def save_model(self, save_path: str):
        """
        Save model to disk
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load model from disk
        
        Args:
            load_path: Path to load model from
        """
        self.model = joblib.load(load_path)
        logger.info(f"Model loaded from {load_path}")
    
    def __repr__(self):
        return f"SVMModel(params={self.params})"


if __name__ == "__main__":
    # Test SVM model
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(500, 50)
    y_train = np.random.choice([0, 1], 500)
    X_test = np.random.randn(100, 50)
    y_test = np.random.choice([0, 1], 100)
    
    # Initialize and train model
    svm_model = SVMModel(kernel='rbf', C=1.0, random_state=42)
    svm_model.train(X_train, y_train)
    
    # Evaluate
    metrics = svm_model.evaluate(X_test, y_test)
    print(f"\nNumber of support vectors: {metrics['n_support_vectors']}")
