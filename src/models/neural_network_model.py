"""
Neural Network Classifier
Deep learning implementation for IoT botnet detection
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class NeuralNetworkModel:
    """Neural Network classifier for botnet detection"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int] = [128, 64, 32],
        activation: str = 'relu',
        dropout_rate: float = 0.3,
        output_activation: str = 'sigmoid',
        n_classes: int = 2
    ):
        """
        Initialize Neural Network model
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            activation: Activation function for hidden layers
            dropout_rate: Dropout rate for regularization
            output_activation: Activation for output layer
            n_classes: Number of output classes
        """
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.output_activation = output_activation
        self.n_classes = n_classes
        
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self) -> keras.Model:
        """Build neural network architecture"""
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=(self.input_dim,)))
        
        # Hidden layers with dropout
        for i, units in enumerate(self.hidden_layers):
            model.add(layers.Dense(
                units,
                activation=self.activation,
                kernel_initializer='he_normal',
                name=f'hidden_{i+1}'
            ))
            model.add(layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}'))
        
        # Output layer
        if self.n_classes == 2:
            # Binary classification
            model.add(layers.Dense(1, activation='sigmoid', name='output'))
        else:
            # Multi-class classification
            model.add(layers.Dense(self.n_classes, activation='softmax', name='output'))
        
        return model
    
    def compile_model(
        self,
        optimizer: str = 'adam',
        learning_rate: float = 0.001
    ):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer name
            learning_rate: Learning rate
        """
        # Create optimizer
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            opt = optimizer
        
        # Compile model
        if self.n_classes == 2:
            loss = 'binary_crossentropy'
            metrics = ['accuracy', keras.metrics.AUC(name='auc')]
        else:
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
        
        self.model.compile(
            optimizer=opt,
            loss=loss,
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 64,
        early_stopping_patience: int = 10,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            verbose: Verbosity level
            
        Returns:
            Training history and metrics
        """
        logger.info("Training Neural Network...")
        logger.info(f"Training samples: {X_train.shape[0]}")
        logger.info(f"Features: {X_train.shape[1]}")
        logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
        
        # Prepare callbacks
        callback_list = []
        
        # Early stopping
        if X_val is not None and early_stopping_patience > 0:
            early_stop = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stop)
        
        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        logger.info("Training completed!")
        
        return self.history.history
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features to predict
            threshold: Classification threshold (for binary classification)
            
        Returns:
            Predicted labels
        """
        predictions = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            # Binary classification
            return (predictions > threshold).astype(int).flatten()
        else:
            # Multi-class classification
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features to predict
            
        Returns:
            Prediction probabilities
        """
        predictions = self.model.predict(X, verbose=0)
        
        if self.n_classes == 2:
            # Binary classification - return probabilities for both classes
            prob_class_1 = predictions.flatten()
            prob_class_0 = 1 - prob_class_1
            return np.column_stack([prob_class_0, prob_class_1])
        else:
            # Multi-class classification
            return predictions
    
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
        logger.info("Evaluating Neural Network model...")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC-AUC
        try:
            if self.n_classes == 2:
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
        except:
            roc_auc = None
        
        # Model evaluation
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)[:2]
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'roc_auc': roc_auc,
            'test_loss': test_loss,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        logger.info(f"Test loss: {test_loss:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        if roc_auc:
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
        
        if return_predictions:
            metrics['predictions'] = y_pred
            metrics['probabilities'] = y_proba
        
        return metrics
    
    def get_model_summary(self):
        """Print model architecture summary"""
        self.model.summary()
    
    def save_model(self, save_path: str):
        """
        Save model to disk
        
        Args:
            save_path: Path to save model
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load model from disk
        
        Args:
            load_path: Path to load model from
        """
        self.model = keras.models.load_model(load_path)
        logger.info(f"Model loaded from {load_path}")
    
    def __repr__(self):
        return f"NeuralNetworkModel(layers={self.hidden_layers}, n_classes={self.n_classes})"


if __name__ == "__main__":
    # Test Neural Network model
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(800, 50)
    y_train = np.random.choice([0, 1], 800)
    X_val = np.random.randn(100, 50)
    y_val = np.random.choice([0, 1], 100)
    X_test = np.random.randn(100, 50)
    y_test = np.random.choice([0, 1], 100)
    
    # Initialize model
    nn_model = NeuralNetworkModel(
        input_dim=50,
        hidden_layers=[64, 32, 16],
        n_classes=2
    )
    
    # Compile
    nn_model.compile_model(learning_rate=0.001)
    
    # Print summary
    nn_model.get_model_summary()
    
    # Train
    history = nn_model.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=32)
    
    # Evaluate
    metrics = nn_model.evaluate(X_test, y_test)
