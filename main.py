"""
Main Execution Script
IoT Botnet Detection via Network Traffic Analysis

Usage:
    python main.py --mode train --models all
    python main.py --mode evaluate
    python main.py --mode predict --input data/test.csv
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import custom modules
from utils.config_loader import Config, create_directories
from utils.logger import setup_logger, ExperimentLogger
from data_processing.data_loader import DataLoader
from data_processing.preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel
from models.neural_network_model import NeuralNetworkModel
from visualization.visualizer import Visualizer

# Setup logger
logger = setup_logger(name="main", log_level="INFO", log_dir="logs")


class IoTBotnetDetector:
    """Main class for IoT Botnet Detection"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize detector
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = Config(config_path)
        create_directories(self.config)
        
        # Initialize components
        self.data_loader = DataLoader(self.config.get('data.raw_data_path'))
        self.preprocessor = DataPreprocessor(
            scaling_method=self.config.get('features.scaling_method'),
            feature_selection=self.config.get('features.feature_selection'),
            n_top_features=self.config.get('features.n_top_features')
        )
        self.visualizer = Visualizer(self.config.get('paths.figures_dir'))
        
        # Initialize experiment logger
        self.exp_logger = ExperimentLogger(
            experiment_name=self.config.get('experiment.experiment_name'),
            log_dir="logs/experiments"
        )
        
        # Data containers
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None
        
        # Model container
        self.models = {}
        
        logger.info("IoT Botnet Detector initialized")
    
    def load_data(self, benign_file: str, attack_files: dict = None):
        """
        Load dataset
        
        Args:
            benign_file: Path to benign traffic file
            attack_files: Dictionary of attack files {name: path}
        """
        logger.info("Loading dataset...")
        
        if attack_files:
            df = self.data_loader.load_benign_and_attacks(benign_file, attack_files)
        else:
            df = self.data_loader.load_csv(benign_file, label='benign')
            df['binary_label'] = 0
        
        # Validate data
        is_valid, issues = self.data_loader.validate_data(df)
        if not is_valid:
            logger.warning(f"Data validation issues: {issues}")
        
        # Separate features and labels
        label_col = 'binary_label' if 'binary_label' in df.columns else 'label'
        self.feature_names = [col for col in df.columns if col not in ['label', 'binary_label']]
        
        X = df[self.feature_names]
        y = df[label_col]
        
        logger.info(f"Loaded {len(df)} samples with {len(self.feature_names)} features")
        logger.info(f"Class distribution:\n{y.value_counts()}")
        
        # Visualize class distribution
        self.visualizer.plot_class_distribution(
            y.values,
            class_names=['Benign', 'Attack'] if len(y.unique()) == 2 else None,
            save_name='class_distribution.png'
        )
        
        return X, y
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series):
        """
        Prepare data for training
        
        Args:
            X: Features
            y: Labels
        """
        logger.info("Preparing data...")
        
        # Handle missing values
        X = self.preprocessor.handle_missing_values(X)
        
        # Split data
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = \
            self.preprocessor.split_data(
                X, y,
                test_size=self.config.get('data.test_size'),
                validation_size=self.config.get('data.validation_size'),
                random_state=self.config.get('data.random_state'),
                stratify=True
            )
        
        # Encode labels
        self.y_train = self.preprocessor.encode_labels(self.y_train, fit=True)
        self.y_val = self.preprocessor.encode_labels(self.y_val, fit=False) if self.y_val is not None else None
        self.y_test = self.preprocessor.encode_labels(self.y_test, fit=False)
        
        # Scale features (fit on train, transform test and val)
        self.X_train, self.X_test = self.preprocessor.scale_features(
            self.X_train, self.X_test
        )
        if self.X_val is not None:
            self.X_val = self.preprocessor.scaler.transform(self.X_val)
        
        # Feature selection (fit on train, transform test and val)
        if self.config.get('features.feature_selection'):
            self.X_train, self.X_test = self.preprocessor.select_features(
                self.X_train, self.y_train, self.X_test
            )
            if self.X_val is not None:
                self.X_val = self.preprocessor.feature_selector.transform(self.X_val)
        
        # Save preprocessor
        self.preprocessor.save_preprocessor('models/preprocessor')
        
        logger.info("Data preparation completed")
    
    def train_models(self, model_names: list = ['all']):
        """
        Train specified models
        
        Args:
            model_names: List of model names to train ('all' for all models)
        """
        if 'all' in model_names:
            model_names = ['random_forest', 'svm', 'neural_network']
        
        logger.info(f"Training models: {model_names}")
        
        for model_name in model_names:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training {model_name.upper()}")
            logger.info(f"{'='*50}")
            
            if model_name == 'random_forest':
                self.train_random_forest()
            elif model_name == 'svm':
                self.train_svm()
            elif model_name == 'neural_network':
                self.train_neural_network()
            else:
                logger.warning(f"Unknown model: {model_name}")
    
    def train_random_forest(self):
        """Train Random Forest model"""
        rf_config = self.config.get('models.random_forest')
        
        model = RandomForestModel(**rf_config)
        self.exp_logger.log_model_info('Random Forest', rf_config)
        
        # Train
        metrics = model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.exp_logger.log_metrics(metrics, phase='training')
        
        # Evaluate
        test_metrics = model.evaluate(self.X_test, self.y_test)
        self.exp_logger.log_metrics(test_metrics, phase='testing')
        
        # Save model
        model.save_model(f"{self.config.get('paths.models_dir')}/random_forest.pkl")
        
        # Visualizations
        self.visualizer.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=['Benign', 'Attack'],
            title='Random Forest - Confusion Matrix',
            save_name='rf_confusion_matrix.png'
        )
        
        # Feature importance — map selected feature indices back to original names
        if self.config.get('features.feature_selection') and \
                self.preprocessor.selected_features is not None and \
                self.feature_names is not None:
            feat_names_for_importance = [
                self.feature_names[i] for i in self.preprocessor.selected_features
            ]
        else:
            feat_names_for_importance = self.feature_names
        top_features = model.get_feature_importance(feat_names_for_importance, top_n=20)
        if top_features:
            self.visualizer.plot_feature_importance(
                top_features,
                title='Random Forest - Top 20 Features',
                save_name='rf_feature_importance.png'
            )
        
        self.models['random_forest'] = {
            'model': model,
            'metrics': test_metrics
        }
    
    def train_svm(self):
        """Train SVM model"""
        svm_config = self.config.get('models.svm')
        
        model = SVMModel(**svm_config)
        self.exp_logger.log_model_info('SVM', svm_config)
        
        # Train
        metrics = model.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.exp_logger.log_metrics(metrics, phase='training')
        
        # Evaluate
        test_metrics = model.evaluate(self.X_test, self.y_test)
        self.exp_logger.log_metrics(test_metrics, phase='testing')
        
        # Save model
        model.save_model(f"{self.config.get('paths.models_dir')}/svm.pkl")
        
        # Visualizations
        self.visualizer.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=['Benign', 'Attack'],
            title='SVM - Confusion Matrix',
            save_name='svm_confusion_matrix.png'
        )
        
        self.models['svm'] = {
            'model': model,
            'metrics': test_metrics
        }
    
    def train_neural_network(self):
        """Train Neural Network model"""
        nn_config = self.config.get('models.neural_network')
        
        # Determine input dimension
        input_dim = self.X_train.shape[1]
        
        model = NeuralNetworkModel(
            input_dim=input_dim,
            hidden_layers=nn_config['hidden_layers'],
            activation=nn_config['activation'],
            dropout_rate=nn_config['dropout_rate'],
            n_classes=2
        )
        
        model.compile_model(
            optimizer=nn_config['optimizer'],
            learning_rate=nn_config['learning_rate']
        )
        
        self.exp_logger.log_model_info('Neural Network', nn_config)
        model.get_model_summary()
        
        # Train
        history = model.train(
            self.X_train, self.y_train,
            self.X_val, self.y_val,
            epochs=nn_config['epochs'],
            batch_size=nn_config['batch_size'],
            early_stopping_patience=nn_config['early_stopping_patience']
        )
        
        # Evaluate
        test_metrics = model.evaluate(self.X_test, self.y_test)
        self.exp_logger.log_metrics(test_metrics, phase='testing')
        
        # Save model
        model.save_model(f"{self.config.get('paths.models_dir')}/neural_network.h5")
        
        # Visualizations
        self.visualizer.plot_training_history(
            history,
            metrics=['loss', 'accuracy'],
            save_name='nn_training_history.png'
        )
        
        self.visualizer.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            class_names=['Benign', 'Attack'],
            title='Neural Network - Confusion Matrix',
            save_name='nn_confusion_matrix.png'
        )
        
        self.models['neural_network'] = {
            'model': model,
            'metrics': test_metrics
        }
    
    def compare_models(self):
        """Compare all trained models"""
        if not self.models:
            logger.warning("No models trained yet")
            return
        
        logger.info("\n" + "="*50)
        logger.info("MODEL COMPARISON")
        logger.info("="*50)
        
        # Collect results
        results = {}
        for name, model_data in self.models.items():
            results[name] = model_data['metrics']
        
        # Plot comparison
        self.visualizer.plot_model_comparison(
            results,
            metrics=['accuracy', 'precision', 'recall', 'f1_score'],
            save_name='model_comparison.png'
        )
        
        # Print summary table
        summary = pd.DataFrame({
            name: {
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') else 'N/A'
            }
            for name, metrics in results.items()
        }).T
        
        logger.info(f"\n{summary}")
        
        # Save to CSV
        summary.to_csv(f"{self.config.get('paths.reports_dir')}/model_comparison.csv")
        
        return summary


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='IoT Botnet Detection - Machine Learning Framework'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'predict'],
        default='train',
        help='Execution mode'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to train (random_forest, svm, neural_network, or all)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--benign',
        type=str,
        default='data/raw/benign_traffic.csv',
        help='Path to benign traffic file'
    )
    parser.add_argument(
        '--attacks',
        type=str,
        nargs='*',
        help='Paths to attack traffic files'
    )
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = IoTBotnetDetector(args.config)
    
    if args.mode == 'train':
        # Load data
        attack_files = {}
        if args.attacks:
            for i, attack_file in enumerate(args.attacks):
                attack_files[f'attack_{i+1}'] = attack_file
        
        X, y = detector.load_data(args.benign, attack_files)
        
        # Prepare data
        detector.prepare_data(X, y)
        
        # Train models
        detector.train_models(args.models)
        
        # Compare models
        detector.compare_models()
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*50)
    
    elif args.mode == 'evaluate':
        logger.info("Evaluation mode - loading saved models...")
        # Add evaluation logic here
    
    elif args.mode == 'predict':
        logger.info("Prediction mode...")
        # Add prediction logic here


if __name__ == "__main__":
    main()
