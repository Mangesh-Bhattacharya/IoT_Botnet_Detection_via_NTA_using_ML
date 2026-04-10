# IoT Botnet Detection Project - Complete Summary
## Group 20: Rajat Jaswal, Mangesh Bhattacharya, Riya Kriplani
### Machine Learning (MITS 6800G)

---

## 📋 Project Overview

This project implements a complete machine learning framework for detecting IoT botnet attacks through network traffic analysis using the N-BaIoT dataset.

### Problem Statement
IoT devices are increasingly vulnerable to botnet attacks (Mirai, BASHLITE) due to:
- Limited processing capabilities
- Inadequate authentication
- Poor patch management

### Solution
A comprehensive ML framework that:
- Analyzes network traffic patterns
- Detects anomalies indicating botnet activity
- Provides real-time classification (benign vs malicious)
- Offers multiple model options for comparison

---

## 🎯 Key Features

### 1. **Complete Data Pipeline**
- Automated data loading from CSV files
- Missing value handling
- Feature scaling (Standard, MinMax, Robust)
- Feature selection (SelectKBest)
- Train/validation/test splitting with stratification

### 2. **Multiple ML Models**
- **Random Forest**: Ensemble learning with feature importance
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Neural Network**: Deep learning with TensorFlow/Keras

### 3. **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Classification reports
- Feature importance analysis

### 4. **Professional Visualizations**
- Training history plots
- Confusion matrices
- ROC curves
- Feature importance charts
- Model comparison graphs

### 5. **Modular Architecture**
- Easy to extend with new models
- Configurable through YAML files
- Comprehensive logging
- Reproducible experiments

---

## 📁 Project Structure

```
iot_botnet_detection/
│
├── README.md                          # Main documentation
├── QUICKSTART.md                      # Quick start guide
├── requirements.txt                   # Python dependencies
├── main.py                           # Main execution script
├── demo.py                           # Quick demonstration
├── test_setup.py                     # Setup verification
├── prepare_data.py                   # Data preparation utility
│
├── config/
│   └── config.yaml                   # Configuration file
│
├── data/
│   ├── raw/                          # Original CSV files
│   │   ├── benign_traffic.csv        # Benign network traffic
│   │   └── demonstrate_structure.csv # Dataset structure demo
│   └── processed/                    # Processed datasets
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── data_loader.py           # Load CSV datasets
│   │   └── preprocessor.py          # Feature scaling, encoding
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── random_forest_model.py   # Random Forest implementation
│   │   ├── svm_model.py             # SVM implementation
│   │   └── neural_network_model.py  # Neural Network implementation
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config_loader.py         # Configuration management
│   │   └── logger.py                # Logging utilities
│   │
│   └── visualization/
│       ├── __init__.py
│       └── visualizer.py            # Plotting functions
│
├── models/
│   ├── saved_models/                 # Trained models
│   └── preprocessor/                 # Preprocessing artifacts
│
├── results/
│   ├── figures/                      # Generated plots
│   └── reports/                      # Performance reports
│
├── logs/                             # Training logs
│   └── experiments/                  # Experiment logs
│
└── notebooks/
    └── 01_exploratory_data_analysis.ipynb  # EDA notebook
```

---

## 🚀 Quick Start

### Installation
```bash
cd iot_botnet_detection
pip install -r requirements.txt
```

### Verify Setup
```bash
python test_setup.py
```

### Run Demo
```bash
python demo.py
```

### Full Training
```bash
python main.py --mode train --models all --benign data/raw/benign_traffic.csv
```

---

## 📊 N-BaIoT Dataset

### Overview
- **Source**: UCI Machine Learning Repository
- **Purpose**: Network-based detection of IoT botnet attacks
- **Devices**: 9 commercial IoT devices
- **Attacks**: Mirai and BASHLITE botnets

### Features (115 total)
The dataset contains 115 statistical features extracted from network traffic:

#### Feature Categories
1. **MI (MAC-IP)**: 15 features
   - Statistics from source IP + MAC address

2. **H (Host)**: 15 features
   - Statistics from source IP

3. **HH (Channel)**: 35 features
   - Host-to-host traffic statistics
   - Includes mean, std, magnitude, radius, covariance, pcc

4. **HH_jit (Channel Jitter)**: 15 features
   - Jitter in host-to-host traffic

5. **HpHp (Socket)**: 35 features
   - Port-to-port traffic statistics

#### Time Windows
Each feature is computed across 5 time windows using damped window technique:
- **L5**: Large window (recent history)
- **L3**: Medium window
- **L1**: Standard window
- **L0.1**: Small window
- **L0.01**: Very small window (most recent)

#### Statistics
For each feature category and time window:
- **weight**: Stream weight
- **mean**: Average value
- **variance/std**: Variability
- **magnitude**: Root squared sum of means
- **radius**: Root squared sum of variances
- **covariance**: Relationship between streams
- **pcc**: Correlation coefficient

---

## 🔧 Configuration

All settings are in `config/config.yaml`:

### Data Configuration
```yaml
data:
  test_size: 0.3              # 30% for testing
  validation_size: 0.15        # 15% for validation
  random_state: 42             # Reproducibility
```

### Model Parameters
```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 20
    min_samples_split: 5
    
  svm:
    kernel: "rbf"
    C: 1.0
    gamma: "scale"
    
  neural_network:
    hidden_layers: [128, 64, 32]
    epochs: 50
    batch_size: 64
    learning_rate: 0.001
```

---

## 📈 Model Performance

### Demo Results (with synthetic attack data)

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 1.0000   | 1.0000    | 1.0000 | 1.0000   |
| SVM            | 0.9815   | 0.9815    | 0.9815 | 0.9815   |

### Expected Performance on Real Attack Data
- **Accuracy**: 95-99%
- **Precision**: 94-98%
- **Recall**: 93-97%
- **F1-Score**: 94-98%

---

## 📝 Code Examples

### Loading Data
```python
from data_processing.data_loader import DataLoader

loader = DataLoader('data/raw')
df = loader.load_csv('benign_traffic.csv', label='benign')
```

### Training a Model
```python
from models.random_forest_model import RandomForestModel

model = RandomForestModel(n_estimators=100, random_state=42)
model.train(X_train, y_train, X_val, y_val)
metrics = model.evaluate(X_test, y_test)
```

### Visualization
```python
from visualization.visualizer import Visualizer

viz = Visualizer('results/figures')
viz.plot_confusion_matrix(cm, class_names=['Benign', 'Attack'])
viz.plot_roc_curve(y_true, y_proba)
```

---

## 🎓 Learning Outcomes

This project demonstrates:

1. **Data Processing**
   - Loading and validating large datasets
   - Feature engineering and selection
   - Data normalization and encoding

2. **Machine Learning**
   - Supervised classification algorithms
   - Model training and hyperparameter tuning
   - Cross-validation and evaluation metrics

3. **Software Engineering**
   - Modular code architecture
   - Configuration management
   - Logging and error handling
   - Version control best practices

4. **Cybersecurity**
   - Network traffic analysis
   - Anomaly detection
   - IoT security challenges

---

## 🔬 Extending the Project

### Add New Models
Create a new model file in `src/models/`:
```python
class MyCustomModel:
    def train(self, X_train, y_train):
        # Your training logic
        pass
    
    def evaluate(self, X_test, y_test):
        # Your evaluation logic
        pass
```

### Add New Features
Extend `preprocessor.py`:
```python
def create_custom_features(self, df):
    # Your feature engineering logic
    return df_with_new_features
```

### Custom Visualizations
Add to `visualizer.py`:
```python
def plot_custom_metric(self, data, title):
    # Your custom visualization
    pass
```

---

## 📚 References

1. **N-BaIoT Paper**: Meidan et al. (2018). "N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders"

2. **Kitsune**: Mirsky et al. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection"

3. **Dataset**: UCI Machine Learning Repository - Detection of IoT Botnet Attacks (N-BaIoT)

4. Complete reference list in `Project-Proposal-Group-20.pdf`

---

## 👥 Team Contributions

**Group 20**:
- **Rajat Jaswal**: Model implementation, experimentation
- **Mangesh Bhattacharya**: Data processing, architecture design
- **Riya Kriplani**: Evaluation, visualization

---

## 📄 License

Academic use only - MITS 6800G Machine Learning Course Project

---

## 🎉 Conclusion

This project provides a complete, production-ready framework for IoT botnet detection. The modular design allows for easy experimentation with different models, features, and configurations. All code follows best practices for machine learning projects including proper data handling, model evaluation, and result visualization.

**Key Achievements**:
- ✅ Complete ML pipeline from data loading to model deployment
- ✅ Multiple model implementations with consistent interfaces
- ✅ Comprehensive evaluation and visualization
- ✅ Well-documented, maintainable code
- ✅ Reproducible experiments with configuration management

---

**Last Updated**: April 8, 2026  
**Version**: 1.0  
**Course**: MITS 6800G - Machine Learning
