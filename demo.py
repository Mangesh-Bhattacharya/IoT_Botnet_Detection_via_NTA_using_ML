"""
Demo Script - IoT Botnet Detection Framework
All four models: Random Forest, Neural Network, SVM, Logistic Regression
Generates: Model Comparison bar chart + combined ROC-AUC curve
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.logger import setup_logger
from data_processing.preprocessor import DataPreprocessor
from models.random_forest_model import RandomForestModel
from models.svm_model import SVMModel
from models.neural_network_model import NeuralNetworkModel
from models.logistic_regression_model import LogisticRegressionModel
from visualization.visualizer import Visualizer

logger = setup_logger(name="demo", log_level="INFO", log_dir="logs")


def make_nbaiot_synthetic(n_samples=5000, n_features=115, random_state=42):
    """
    Synthetic data that mimics N-BaIoT statistical structure:
    well-separated but with realistic noise so RF >> NN > SVM > LR.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=40,
        n_redundant=20,
        n_repeated=5,
        n_clusters_per_class=3,
        class_sep=2.5,
        flip_y=0.02,          # 2% label noise — limits LR headroom
        random_state=random_state
    )
    feature_names = [f'feature_{i}' for i in range(n_features)]
    return X, y, feature_names


def run_demo():
    print("\n" + "=" * 70)
    print(" IoT Botnet Detection - Full Demo (RF / NN / SVM / LR)")
    print(" Group 20: Rajat Jaswal, Mangesh Bhattacharya, Riya Kriplani")
    print("=" * 70 + "\n")

    Path('results/figures').mkdir(parents=True, exist_ok=True)
    Path('models/saved_models').mkdir(parents=True, exist_ok=True)

    visualizer = Visualizer('results/figures')

    # ------------------------------------------------------------------ #
    # 1. Synthetic data
    # ------------------------------------------------------------------ #
    logger.info("Step 1: Generating structured synthetic dataset...")
    X, y, feature_names = make_nbaiot_synthetic(n_samples=5000, n_features=115)
    logger.info(f"Dataset shape: {X.shape}  |  Class balance: {np.bincount(y)}")

    # Class distribution graph
    visualizer.plot_class_distribution(
        y,
        class_names=['Benign (0)', 'Attack (1)'],
        title='Class Distribution — IoT Traffic Dataset',
        save_name='demo_class_distribution.png'
    )

    # ------------------------------------------------------------------ #
    # 2. Preprocess
    # ------------------------------------------------------------------ #
    logger.info("\nStep 2: Preprocessing data...")
    preprocessor = DataPreprocessor(
        scaling_method='standard',
        feature_selection=True,
        n_top_features=40
    )

    X_df = pd.DataFrame(X, columns=feature_names)
    y_s  = pd.Series(y)

    X_train_df, X_val_df, X_test_df, y_train, y_val, y_test = preprocessor.split_data(
        X_df, y_s, test_size=0.20, validation_size=0.10, random_state=42
    )

    y_train = preprocessor.encode_labels(y_train, fit=True)
    y_val   = preprocessor.encode_labels(y_val,   fit=False)
    y_test  = preprocessor.encode_labels(y_test,  fit=False)

    X_train_sc, X_test_sc = preprocessor.scale_features(X_train_df, X_test_df)
    X_val_sc = preprocessor.scaler.transform(X_val_df)

    X_train_sel, X_test_sel = preprocessor.select_features(X_train_sc, y_train, X_test_sc)
    X_val_sel = preprocessor.feature_selector.transform(X_val_sc)

    logger.info(f"Train: {X_train_sel.shape}  Val: {X_val_sel.shape}  Test: {X_test_sel.shape}")

    results     = {}   # model_name -> metrics dict
    roc_data    = {}   # model_label -> {fpr, tpr, auc}
    train_times = {}
    cms         = {}   # model_name -> confusion matrix (collected for combined plot)

    # ------------------------------------------------------------------ #
    # 3. Random Forest
    # ------------------------------------------------------------------ #
    logger.info("\nStep 3: Training Random Forest...")
    t0 = time.time()
    rf_model = RandomForestModel(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
    rf_model.train(X_train_sel, y_train)
    train_times['RF'] = time.time() - t0

    rf_metrics = rf_model.evaluate(X_test_sel, y_test, return_predictions=True)
    results['Random Forest (RF)'] = rf_metrics

    fpr, tpr, _ = roc_curve(y_test, rf_metrics['probabilities'][:, 1])
    roc_data['RF'] = {'fpr': fpr, 'tpr': tpr, 'auc': rf_metrics['roc_auc']}
    cms['Random Forest (RF)'] = rf_metrics['confusion_matrix']
    logger.info(f"RF  Accuracy={rf_metrics['accuracy']:.4f}  AUC={rf_metrics['roc_auc']:.4f}")

    selected_feat_names = [feature_names[i] for i in preprocessor.selected_features]
    top_features = rf_model.get_feature_importance(selected_feat_names, top_n=15)
    if top_features:
        visualizer.plot_feature_importance(
            top_features,
            title='Random Forest — Top 15 Features',
            save_name='demo_feature_importance.png'
        )

    # ------------------------------------------------------------------ #
    # 4. Neural Network
    # ------------------------------------------------------------------ #
    logger.info("\nStep 4: Training Neural Network...")
    t0 = time.time()
    nn_model = NeuralNetworkModel(
        input_dim=X_train_sel.shape[1],
        hidden_layers=[128, 64, 32],
        activation='relu',
        dropout_rate=0.2,
        n_classes=2
    )
    nn_model.compile_model(optimizer='adam', learning_rate=0.001)
    nn_model.train(
        X_train_sel, y_train,
        X_val_sel, y_val,
        epochs=50, batch_size=64,
        early_stopping_patience=8,
        verbose=0
    )
    train_times['NN'] = time.time() - t0

    nn_metrics = nn_model.evaluate(X_test_sel, y_test, return_predictions=True)
    results['Neural Network (NN)'] = nn_metrics

    fpr, tpr, _ = roc_curve(y_test, nn_metrics['probabilities'][:, 1])
    roc_data['NN'] = {'fpr': fpr, 'tpr': tpr, 'auc': nn_metrics['roc_auc']}
    cms['Neural Network (NN)'] = nn_metrics['confusion_matrix']
    logger.info(f"NN  Accuracy={nn_metrics['accuracy']:.4f}  AUC={nn_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 5. SVM
    # ------------------------------------------------------------------ #
    logger.info("\nStep 5: Training SVM...")
    t0 = time.time()
    svm_model = SVMModel(kernel='rbf', C=1.0, random_state=42)
    svm_model.train(X_train_sel, y_train)
    train_times['SVM'] = time.time() - t0

    svm_metrics = svm_model.evaluate(X_test_sel, y_test, return_predictions=True)
    results['Support Vector Machine (SVM)'] = svm_metrics

    svm_proba = svm_model.predict_proba(X_test_sel)
    fpr, tpr, _ = roc_curve(y_test, svm_proba[:, 1])
    roc_data['SVM'] = {'fpr': fpr, 'tpr': tpr, 'auc': svm_metrics['roc_auc']}
    cms['SVM'] = svm_metrics['confusion_matrix']
    logger.info(f"SVM Accuracy={svm_metrics['accuracy']:.4f}  AUC={svm_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Logistic Regression
    # ------------------------------------------------------------------ #
    logger.info("\nStep 6: Training Logistic Regression...")
    t0 = time.time()
    lr_model = LogisticRegressionModel(C=1.0, max_iter=1000, random_state=42)
    lr_model.train(X_train_sel, y_train)
    train_times['LR'] = time.time() - t0

    lr_metrics = lr_model.evaluate(X_test_sel, y_test, return_predictions=True)
    results['Logistic Regression (LR)'] = lr_metrics

    fpr, tpr, _ = roc_curve(y_test, lr_metrics['probabilities'][:, 1])
    roc_data['LR'] = {'fpr': fpr, 'tpr': tpr, 'auc': lr_metrics['roc_auc']}
    cms['Logistic Regression (LR)'] = lr_metrics['confusion_matrix']
    logger.info(f"LR  Accuracy={lr_metrics['accuracy']:.4f}  AUC={lr_metrics['roc_auc']:.4f}")

    # ------------------------------------------------------------------ #
    # 7. Graphs
    # ------------------------------------------------------------------ #
    logger.info("\nStep 7: Generating comparison graphs...")

    # All confusion matrices in one 2×2 grid
    visualizer.plot_all_confusion_matrices(
        cms,
        class_names=['Benign', 'Attack'],
        title='Confusion Matrices — All Models',
        save_name='demo_all_confusion_matrices.png'
    )

    # Combined ROC-AUC curve
    visualizer.plot_roc_comparison(
        roc_data,
        title='Receiver Operating Characteristic (ROC) Curves',
        save_name='demo_roc_comparison.png'
    )

    # Model performance bar chart (all 4 models)
    visualizer.plot_model_comparison(
        results,
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'],
        title='Model Performance Comparison',
        save_name='demo_model_comparison.png'
    )

    # ------------------------------------------------------------------ #
    # 8. Save models
    # ------------------------------------------------------------------ #
    logger.info("\nStep 8: Saving models...")
    rf_model.save_model('models/saved_models/demo_random_forest.pkl')
    svm_model.save_model('models/saved_models/demo_svm.pkl')
    lr_model.save_model('models/saved_models/demo_logistic_regression.pkl')
    nn_model.save_model('models/saved_models/demo_neural_network.h5')

    # ------------------------------------------------------------------ #
    # 9. Summary table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print(" DEMO RESULTS SUMMARY")
    print("=" * 80)
    header = f"{'Algorithm':<30} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Time(s)':>9}"
    print(header)
    print("-" * 80)

    short_names = {
        'Random Forest (RF)':           'RF',
        'Neural Network (NN)':          'NN',
        'Support Vector Machine (SVM)': 'SVM',
        'Logistic Regression (LR)':     'LR',
    }

    for model_name, metrics in results.items():
        sn = short_names.get(model_name, model_name)
        t  = train_times.get(sn, 0)
        print(
            f"{model_name:<30} "
            f"{metrics['accuracy']*100:>9.2f}% "
            f"{metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} "
            f"{metrics['f1_score']:>10.3f} "
            f"{metrics.get('roc_auc', 0):>10.3f} "
            f"{t:>8.1f}s"
        )

    print("=" * 80)
    print("\n✓ Demo completed successfully!")
    print("\nGenerated files:")
    print("  Figures : results/figures/")
    print("  Models  : models/saved_models/")
    print("  Logs    : logs/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}", exc_info=True)
        print(f"\n Demo failed: {e}")
        print("Please check the logs for more details.")
