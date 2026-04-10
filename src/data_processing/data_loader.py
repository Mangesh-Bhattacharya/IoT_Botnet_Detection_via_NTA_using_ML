"""
Data Loader
Load and prepare IoT botnet detection datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage N-BaIoT dataset"""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory containing raw CSV files
        """
        self.data_dir = Path(data_dir)
        self.feature_names = None
        
    def load_csv(self, file_path: str, label: str = None) -> pd.DataFrame:
        """
        Load a single CSV file
        
        Args:
            file_path: Path to CSV file
            label: Label to assign to this data (e.g., 'benign', 'mirai', 'bashlite')
            
        Returns:
            DataFrame with features and label
        """
        logger.info(f"Loading file: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Store feature names
        if self.feature_names is None:
            self.feature_names = list(df.columns)
        
        # Add label column if provided
        if label is not None:
            df['label'] = label
            
        logger.info(f"Loaded {len(df)} samples from {Path(file_path).name}")
        
        return df
    
    def load_multiple_files(
        self, 
        file_label_pairs: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """
        Load multiple CSV files and combine them
        
        Args:
            file_label_pairs: List of (file_path, label) tuples
            
        Returns:
            Combined DataFrame
        """
        dfs = []
        
        for file_path, label in file_label_pairs:
            df = self.load_csv(file_path, label)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Combined dataset shape: {combined_df.shape}")
        
        return combined_df
    
    def load_benign_and_attacks(
        self,
        benign_file: str,
        attack_files: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Load benign traffic and attack files
        
        Args:
            benign_file: Path to benign traffic file
            attack_files: Dictionary of {attack_name: file_path}
            
        Returns:
            Combined DataFrame with binary labels (0=benign, 1=attack)
        """
        # Load benign traffic
        benign_df = self.load_csv(benign_file, label='benign')
        
        # Load attack traffic
        attack_dfs = []
        for attack_name, file_path in attack_files.items():
            attack_df = self.load_csv(file_path, label=attack_name)
            attack_dfs.append(attack_df)
        
        # Combine all data
        if attack_dfs:
            all_attacks = pd.concat(attack_dfs, ignore_index=True)
            combined_df = pd.concat([benign_df, all_attacks], ignore_index=True)
        else:
            combined_df = benign_df
        
        # Create binary label (0=benign, 1=attack)
        combined_df['binary_label'] = (combined_df['label'] != 'benign').astype(int)
        
        logger.info(f"Dataset composition:")
        logger.info(f"  Benign: {len(benign_df)} samples")
        if attack_dfs:
            logger.info(f"  Attacks: {len(all_attacks)} samples")
        logger.info(f"  Total: {len(combined_df)} samples")
        
        return combined_df
    
    def get_feature_info(self) -> Dict:
        """
        Get information about dataset features
        
        Returns:
            Dictionary with feature statistics
        """
        if self.feature_names is None:
            return {}
        
        # Parse feature names to understand structure
        feature_info = {
            'total_features': len(self.feature_names),
            'MI_features': len([f for f in self.feature_names if f.startswith('MI_')]),
            'H_features': len([f for f in self.feature_names if f.startswith('H_') and not f.startswith('HH_')]),
            'HH_features': len([f for f in self.feature_names if f.startswith('HH_') and not f.startswith('HH_jit')]),
            'HH_jit_features': len([f for f in self.feature_names if f.startswith('HH_jit')]),
            'HpHp_features': len([f for f in self.feature_names if f.startswith('HpHp_')]),
        }
        
        return feature_info
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate loaded data
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            issues.append(f"Missing values found in {missing[missing > 0].to_dict()}")
        
        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                issues.append(f"Infinite values found in column: {col}")
        
        # Check feature count
        expected_features = 115  # N-BaIoT has 115 features
        actual_features = len([col for col in df.columns if col not in ['label', 'binary_label']])
        if actual_features != expected_features:
            issues.append(f"Expected {expected_features} features, found {actual_features}")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues


def create_sample_data(output_path: str, n_samples: int = 1000):
    """
    Create sample data for testing (when real data is not available)
    
    Args:
        output_path: Where to save sample CSV
        n_samples: Number of samples to generate
    """
    logger.info(f"Creating sample data with {n_samples} samples")
    
    # Create 115 random features
    np.random.seed(42)
    n_features = 115
    
    data = np.random.randn(n_samples, n_features)
    
    # Create feature names matching N-BaIoT structure
    feature_names = []
    for prefix in ['MI_dir', 'H', 'HH', 'HH_jit', 'HpHp']:
        for time_frame in ['L5', 'L3', 'L1', 'L0.1', 'L0.01']:
            for stat in ['weight', 'mean', 'variance', 'std', 'magnitude', 'radius', 'covariance', 'pcc']:
                if len(feature_names) < n_features:
                    feature_names.append(f"{prefix}_{time_frame}_{stat}")
    
    # Adjust to exactly 115 features
    feature_names = feature_names[:n_features]
    
    df = pd.DataFrame(data, columns=feature_names)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data saved to: {output_path}")


if __name__ == "__main__":
    # Test data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = DataLoader()
    
    # Create sample data for testing
    create_sample_data("data/raw/sample_benign.csv", n_samples=500)
    create_sample_data("data/raw/sample_attack.csv", n_samples=300)
