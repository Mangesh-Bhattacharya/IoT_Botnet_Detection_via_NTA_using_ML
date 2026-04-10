# IoT Botnet Detection via Network Traffic Analysis Using Machine Learning

## Project Overview
This project implements a machine learning framework to detect IoT botnet attacks by analyzing network traffic patterns. Using the N-BaIoT dataset, we develop and compare multiple ML models to distinguish between benign and malicious traffic from IoT devices.

## Team Members
- Rajat Jaswal
- Mangesh Bhattacharya
- Riya Kriplani
- **Group 20** - Machine Learning (MITS 6800G)

## Problem Statement
IoT devices are increasingly vulnerable to botnet attacks (Mirai, BASHLITE) due to limited processing capabilities, inadequate authentication, and poor patch management. This project addresses the challenge of detecting these attacks through network traffic analysis using machine learning techniques.

## Dataset
**N-BaIoT Dataset** (Network-based Detection of IoT Botnet Attacks)
- **Source**: UCI Machine Learning Repository
- **Features**: 115 statistical features extracted from network traffic
- **Classes**: Benign and 10 types of botnet attacks
- **Devices**: 9 commercial IoT devices
- **Attacks**: Mirai and BASHLITE botnets

### Feature Categories
- **H (Host)**: Statistics from source IP
- **MI (MAC-IP)**: Statistics from source IP + MAC
- **HH (Channel)**: Host-to-host traffic statistics
- **HH_jit (Channel Jitter)**: Jitter in host-to-host traffic
- **HpHp (Socket)**: Port-to-port traffic statistics

### Time Frames
- L5, L3, L1, L0.1, L0.01 (decay factors for damped window)

## Project Structure
```
iot_botnet_detection/
│
├── data/
│   ├── raw/                    # Original CSV files
│   └── processed/              # Processed datasets
│
├── src/
│   ├── data_processing/        # Data loading and preprocessing
│   ├── models/                 # ML model implementations
│   ├── utils/                  # Utility functions
│   └── visualization/          # Plotting and visualization
│
├── notebooks/                  # Jupyter notebooks for exploration
├── models/                     # Saved trained models
├── results/                    # Figures and reports
├── config/                     # Configuration files
├── logs/                       # Training logs
└── requirements.txt            # Python dependencies
```

## Machine Learning Techniques

### Implemented Models
1. **Random Forest Classifier**
2. **Support Vector Machine (SVM)**
3. **Logistic Regression**
4. **Neural Networks (Deep Learning)**
5. **Ensemble Methods**

### Evaluation Metrics
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup
```bash
# Clone or navigate to project directory
cd iot_botnet_detection

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
```bash
python src/data_processing/prepare_data.py
```

### 2. Exploratory Data Analysis
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 3. Train Models
```bash
# Train all models
python main.py --mode train --models all

# Train specific model
python main.py --mode train --models random_forest

# With custom configuration
python main.py --mode train --config config/custom_config.yaml
```

### 4. Evaluate Models
```bash
python main.py --mode evaluate
```

### 5. Make Predictions
```bash
python main.py --mode predict --input data/raw/test_data.csv
```

## Key Features

- **Automated Data Pipeline**: Preprocessing, feature scaling, and train-test splitting
- **Multiple ML Models**: Compare performance across different algorithms
- **Hyperparameter Tuning**: Grid search and cross-validation
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Modular Design**: Easy to extend with new models or features
- **Logging**: Track experiments and model performance
- **Reproducibility**: Fixed random seeds and configuration files

## Results

Model performance will be saved in `results/reports/` directory with:
- Classification reports
- Confusion matrices
- ROC curves
- Feature importance plots
- Performance comparison charts

## References

1. Meidan et al. (2018). "N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders"
2. Mirsky et al. (2018). "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection"
3. See `Project-Proposal-Group-20.pdf` for complete reference list

## License
Academic use only - MITS 6800G Machine Learning Course Project

## Contact
For questions or contributions, please contact team members through the university portal.
