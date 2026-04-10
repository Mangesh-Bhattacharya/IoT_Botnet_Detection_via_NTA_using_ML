"""
Random Forest Classifier
Implementation of Random Forest for IoT botnet detection
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class RandomForestModel:
    """Random Forest classifier for botnet detection"""
    
    def __init__(self, **params):
        """
        Initialize Random Forest model
        
        Args:
            **params: Parameters for RandomForestClassifier
        """
        self.model = RandomForestClassifier(**params)
        self.feature_importance = None
        self.params = params
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary of training metrics
        """
        logger.info("Training Random Forest model...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
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
        logger.info("Evaluating Random Forest model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC-AUC (for binary classification)
        try:
            if len(np.unique(y_test)) == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except:
            roc_auc = None
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        if roc_auc:
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        if return_predictions:
            metrics['predictions'] = y_pred
            metrics['probabilities'] = y_proba
        
        return metrics
    
    def get_feature_importance(self, feature_names=None, top_n: int = 20):
        """
        Get feature importance
        
        Args:
            feature_names: Names of features
            top_n: Number of top features to return
            
        Returns:
            Dictionary or sorted list of feature importance
        """
        if self.feature_importance is None:
            logger.warning("Model not trained yet")
            return None
        
        if feature_names is not None:
            importance_dict = dict(zip(feature_names, self.feature_importance))
            sorted_importance = sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return sorted_importance[:top_n]
        else:
            indices = np.argsort(self.feature_importance)[::-1]
            return [(i, self.feature_importance[i]) for i in indices[:top_n]]
    
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
        self.feature_importance = self.model.feature_importances_
        logger.info(f"Model loaded from {load_path}")
    
    def __repr__(self):
        return f"RandomForestModel(params={self.params})"


if __name__ == "__main__":
    # Test Random Forest model
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(800, 50)
    y_train = np.random.choice([0, 1], 800)
    X_test = np.random.randn(200, 50)
    y_test = np.random.choice([0, 1], 200)
    
    # Initialize and train model
    rf_model = RandomForestModel(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.train(X_train, y_train)
    
    # Evaluate
    metrics = rf_model.evaluate(X_test, y_test)
    
    # Get feature importance
    top_features = rf_model.get_feature_importance(top_n=10)
    print("\nTop 10 Features:")
    for idx, (feat_idx, importance) in enumerate(top_features, 1):
        print(f"{idx}. Feature {feat_idx}: {importance:.4f}")
