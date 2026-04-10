# Quick Start Guide
## IoT Botnet Detection Project - Group 20

This guide will help you get started with the IoT Botnet Detection project quickly.

## Prerequisites

Make sure you have Python 3.8+ installed on your system.

## Installation Steps

### 1. Navigate to Project Directory
```bash
cd iot_botnet_detection
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- NumPy, Pandas, Scikit-learn
- TensorFlow/Keras for deep learning
- Matplotlib, Seaborn for visualization
- And more...

### 3. Verify Installation
```bash
python test_setup.py
```

## Running the Project

### Option 1: Train All Models (Recommended for First Run)
```bash
python main.py --mode train --models all --benign data/raw/benign_traffic.csv
```

This will:
- Load the benign traffic data
- Preprocess and split the data
- Train Random Forest, SVM, and Neural Network models
- Generate visualizations and performance reports
- Save trained models to `models/saved_models/`

### Option 2: Train Specific Models
```bash
# Train only Random Forest
python main.py --mode train --models random_forest --benign data/raw/benign_traffic.csv

# Train Random Forest and SVM
python main.py --mode train --models random_forest svm --benign data/raw/benign_traffic.csv
```

### Option 3: Train with Attack Data
If you have attack traffic files:
```bash
python main.py --mode train --models all \
    --benign data/raw/benign_traffic.csv \
    --attacks data/raw/attack1.csv data/raw/attack2.csv
```

## Exploring the Data

### Using Jupyter Notebook
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

This notebook contains:
- Data loading and inspection
- Feature distribution analysis
- Correlation analysis
- Summary statistics
- Data quality checks

## Project Structure Overview

```
iot_botnet_detection/
├── data/
│   ├── raw/                       # Your CSV files
│   │   └── benign_traffic.csv     # Benign network traffic
│   └── processed/                 # Processed datasets
│
├── src/
│   ├── data_processing/           # Data loading and preprocessing
│   │   ├── data_loader.py         # Load CSV files
│   │   └── preprocessor.py        # Scale, encode, split data
│   ├── models/                    # ML model implementations
│   │   ├── random_forest_model.py
│   │   ├── svm_model.py
│   │   └── neural_network_model.py
│   ├── utils/                     # Utility functions
│   │   ├── config_loader.py       # Configuration management
│   │   └── logger.py              # Logging utilities
│   └── visualization/             # Plotting and visualization
│       └── visualizer.py
│
├── models/                        # Saved trained models
│   └── saved_models/
│
├── results/                       # Output results
│   ├── figures/                   # Generated plots
│   └── reports/                   # Performance reports
│
├── config/                        # Configuration files
│   └── config.yaml                # Main configuration
│
├── logs/                          # Training logs
├── notebooks/                     # Jupyter notebooks
├── main.py                        # Main execution script
└── requirements.txt               # Python dependencies
```

## Understanding the Output

After training, you'll find:

### 1. Trained Models
Located in `models/saved_models/`:
- `random_forest.pkl` - Random Forest model
- `svm.pkl` - Support Vector Machine model
- `neural_network.h5` - Neural Network model
- `models/preprocessor/` - Data preprocessing artifacts

### 2. Visualizations
Located in `results/figures/`:
- `class_distribution.png` - Distribution of benign vs attack samples
- `rf_confusion_matrix.png` - Random Forest confusion matrix
- `svm_confusion_matrix.png` - SVM confusion matrix
- `nn_confusion_matrix.png` - Neural Network confusion matrix
- `nn_training_history.png` - Neural Network training curves
- `rf_feature_importance.png` - Most important features
- `model_comparison.png` - Performance comparison across models

### 3. Reports
Located in `results/reports/`:
- `model_comparison.csv` - Numerical comparison of all models

### 4. Logs
Located in `logs/`:
- Detailed training logs with timestamps
- Experiment logs with metrics

## Configuration

Edit `config/config.yaml` to customize:

### Data Settings
```yaml
data:
  test_size: 0.3        # 30% for testing
  validation_size: 0.15  # 15% for validation
  random_state: 42       # For reproducibility
```

### Model Parameters
```yaml
models:
  random_forest:
    n_estimators: 100
    max_depth: 20
    
  svm:
    kernel: "rbf"
    C: 1.0
    
  neural_network:
    hidden_layers: [128, 64, 32]
    epochs: 50
    batch_size: 64
```

## Expected Results

With the benign traffic data, you should see:

### Random Forest
- **Training Time**: ~10-30 seconds
- **Expected Accuracy**: 95-99% (on synthetic or clean data)
- **Feature Importance**: Available for analysis

### SVM
- **Training Time**: ~30-60 seconds (depends on data size)
- **Expected Accuracy**: 90-98%
- **Support Vectors**: Information about decision boundary

### Neural Network
- **Training Time**: ~2-5 minutes
- **Expected Accuracy**: 92-99%
- **Training Curves**: Loss and accuracy over epochs

## Common Issues and Solutions

### Issue 1: Out of Memory
**Solution**: Reduce batch size or use fewer features
```yaml
neural_network:
  batch_size: 32  # Reduce from 64
  
features:
  feature_selection: true
  n_top_features: 30  # Reduce from 50
```

### Issue 2: SVM Training Too Slow
**Solution**: Use linear kernel or reduce dataset size
```yaml
svm:
  kernel: "linear"  # Faster than "rbf"
```

### Issue 3: Neural Network Not Converging
**Solution**: Adjust learning rate and epochs
```yaml
neural_network:
  learning_rate: 0.0001  # Reduce learning rate
  epochs: 100            # Increase epochs
```

## Next Steps

1. **Experiment with Hyperparameters**: Modify `config/config.yaml`
2. **Add More Attack Data**: Include different attack types
3. **Feature Engineering**: Create new features from existing ones
4. **Ensemble Methods**: Combine multiple models
5. **Deploy Models**: Use trained models for real-time detection

## Getting Help

- Check the logs in `logs/` directory
- Review the dataset description: `N_BaIoT_dataset_description.txt`
- Read the project proposal: `Project-Proposal-Group-20.pdf`

## Performance Metrics Explained

- **Accuracy**: Percentage of correct predictions
- **Precision**: Of all predicted attacks, how many were actually attacks?
- **Recall**: Of all actual attacks, how many did we detect?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (higher is better)

## Tips for Best Results

1. **Start Simple**: Train one model first to understand the pipeline
2. **Check Logs**: Always review logs for errors or warnings
3. **Visualize Results**: Look at confusion matrices and ROC curves
4. **Compare Models**: Use the comparison chart to pick the best model
5. **Save Your Work**: Models are automatically saved after training

## Happy Training! 🚀

For questions or issues, refer to the main README.md or check the documentation in the code files.
