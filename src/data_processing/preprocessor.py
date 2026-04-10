"""
Data Preprocessing
Feature scaling, encoding, and data preparation for ML models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from typing import Tuple, Optional
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess data for machine learning models"""
    
    def __init__(
        self,
        scaling_method: str = 'standard',
        feature_selection: bool = False,
        n_top_features: int = 50
    ):
        """
        Initialize preprocessor
        
        Args:
            scaling_method: Method for feature scaling ('standard', 'minmax', 'robust')
            feature_selection: Whether to perform feature selection
            n_top_features: Number of top features to select
        """
        self.scaling_method = scaling_method
        self.feature_selection = feature_selection
        self.n_top_features = n_top_features
        
        # Initialize scalers and encoders
        self.scaler = self._get_scaler()
        self.label_encoder = LabelEncoder()
        self.feature_selector = None
        self.selected_features = None
        
    def _get_scaler(self):
        """Get appropriate scaler based on method"""
        scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
        return scalers.get(self.scaling_method, StandardScaler())
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'mean'
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values")
            
            if strategy == 'drop':
                df = df.dropna()
            elif strategy in ['mean', 'median']:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if strategy == 'mean':
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                else:
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        return df
    
    def handle_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect and handle outliers
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outliers handled
        """
        logger.info(f"Handling outliers with method: {method}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['label', 'binary_label']]
        
        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
                
        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df.loc[z_scores > threshold, col] = df[col].median()
        
        return df
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using the specified scaler
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            
        Returns:
            Tuple of scaled (X_train, X_test)
        """
        logger.info(f"Scaling features using {self.scaling_method} scaler")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Transform test data if provided
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray = None,
        method: str = 'f_classif'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Select top features based on statistical tests
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            method: Feature selection method ('f_classif', 'mutual_info')
            
        Returns:
            Tuple of (X_train_selected, X_test_selected)
        """
        if not self.feature_selection:
            return X_train, X_test
        
        logger.info(f"Selecting top {self.n_top_features} features using {method}")
        
        # Choose scoring function
        score_func = f_classif if method == 'f_classif' else mutual_info_classif
        
        # Initialize feature selector
        self.feature_selector = SelectKBest(score_func=score_func, k=self.n_top_features)
        
        # Fit and transform
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        # Get selected feature indices
        self.selected_features = self.feature_selector.get_support(indices=True)
        
        # Transform test data if provided
        X_test_selected = None
        if X_test is not None:
            X_test_selected = self.feature_selector.transform(X_test)
        
        logger.info(f"Selected features: {len(self.selected_features)}")
        
        return X_train_selected, X_test_selected
    
    def encode_labels(self, y: pd.Series, fit: bool = True) -> np.ndarray:
        """
        Encode categorical labels to numeric
        
        Args:
            y: Labels to encode
            fit: Whether to fit the encoder
            
        Returns:
            Encoded labels
        """
        if fit:
            return self.label_encoder.fit_transform(y)
        else:
            return self.label_encoder.transform(y)
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.3,
        validation_size: float = 0.15,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple:
        """
        Split data into train, validation, and test sets
        
        Args:
            X: Features
            y: Labels
            test_size: Proportion of test set
            validation_size: Proportion of validation set (from training data)
            random_state: Random seed
            stratify: Whether to use stratified sampling
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info(f"Splitting data: test_size={test_size}, validation_size={validation_size}")
        
        stratify_param = y if stratify else None
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Second split: separate validation from training
        if validation_size > 0:
            val_size_adjusted = validation_size / (1 - test_size)
            stratify_temp = y_temp if stratify else None
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size_adjusted,
                random_state=random_state,
                stratify=stratify_temp
            )
        else:
            X_train, y_train = X_temp, y_temp
            X_val, y_val = None, None
        
        logger.info(f"Train set: {X_train.shape}")
        if X_val is not None:
            logger.info(f"Validation set: {X_val.shape}")
        logger.info(f"Test set: {X_test.shape}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_class_distribution(self, y: pd.Series) -> pd.Series:
        """Get class distribution"""
        return y.value_counts()
    
    def save_preprocessor(self, save_dir: str):
        """
        Save preprocessor state
        
        Args:
            save_dir: Directory to save preprocessor
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save scaler
        joblib.dump(self.scaler, save_path / 'scaler.pkl')
        
        # Save label encoder
        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')
        
        # Save feature selector if used
        if self.feature_selector is not None:
            joblib.dump(self.feature_selector, save_path / 'feature_selector.pkl')
            np.save(save_path / 'selected_features.npy', self.selected_features)
        
        logger.info(f"Preprocessor saved to {save_dir}")
    
    def load_preprocessor(self, load_dir: str):
        """
        Load preprocessor state
        
        Args:
            load_dir: Directory to load from
        """
        load_path = Path(load_dir)
        
        # Load scaler
        self.scaler = joblib.load(load_path / 'scaler.pkl')
        
        # Load label encoder
        self.label_encoder = joblib.load(load_path / 'label_encoder.pkl')
        
        # Load feature selector if exists
        selector_path = load_path / 'feature_selector.pkl'
        if selector_path.exists():
            self.feature_selector = joblib.load(selector_path)
            self.selected_features = np.load(load_path / 'selected_features.npy')
        
        logger.info(f"Preprocessor loaded from {load_dir}")


if __name__ == "__main__":
    # Test preprocessor
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(1000, 115))
    y = pd.Series(np.random.choice(['benign', 'attack'], 1000))
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(feature_selection=True, n_top_features=50)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(X, y)
    
    # Encode labels
    y_train_encoded = preprocessor.encode_labels(y_train)
    
    # Scale features
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Select features
    X_train_selected, X_test_selected = preprocessor.select_features(
        X_train_scaled, y_train_encoded, X_test_scaled
    )
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Selected features: {X_train_selected.shape[1]}")
