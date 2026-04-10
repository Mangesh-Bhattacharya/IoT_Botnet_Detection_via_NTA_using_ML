"""
Simple Training Example
A straightforward example of training a model on IoT botnet data
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from data_processing.data_loader import DataLoader
from data_processing.preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from visualization.visualizer import Visualizer

print("="*60)
print("IoT Botnet Detection - Simple Training Example")
print("="*60)

# 1. Load data
print("\n1. Loading data...")
loader = DataLoader('data/raw')
df = loader.load_csv('data/raw/benign_traffic.csv', label='benign')
print(f"   Loaded {len(df)} samples with {len(df.columns)-1} features")

# 2. Prepare data
print("\n2. Preparing data...")
X = df.drop('label', axis=1)
y = df['label']

preprocessor = DataPreprocessor()
X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
    X, y, test_size=0.2, validation_size=0, random_state=42
)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# 3. Encode and scale
print("\n3. Encoding and scaling...")
y_train = preprocessor.encode_labels(y_train, fit=True)
y_test = preprocessor.encode_labels(y_test, fit=False)
X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

# 4. Train model
print("\n4. Training Random Forest...")
model = RandomForestModel(n_estimators=50, max_depth=10, random_state=42)
model.train(X_train_scaled, y_train)

# 5. Evaluate
print("\n5. Evaluating on test set...")
metrics = model.evaluate(X_test_scaled, y_test)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
print("="*60)

# 6. Save model
print("\n6. Saving model...")
model.save_model('models/saved_models/simple_example_rf.pkl')
print("   Model saved!")

print("\n✓ Training complete!")
print("="*60)
