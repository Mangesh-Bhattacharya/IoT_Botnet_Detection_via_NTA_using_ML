"""
Visualization Utilities
Plotting and visualization functions for model evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from pathlib import Path
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """Visualization tools for IoT botnet detection results"""
    
    def __init__(self, save_dir: str = "results/figures"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def _draw_confusion_matrix(self, ax, cm: np.ndarray, class_names: List[str], title: str):
        """
        Internal helper: draw a single confusion matrix onto an existing Axes.
        Each cell shows count and row-normalised percentage.
        Diagonal (correct) cells use a green tint; off-diagonal (errors) use red.
        """
        n = cm.shape[0]
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sums > 0, cm.astype(float) / row_sums * 100, 0.0)

        # Build colour array: green for TP/TN, red for FP/FN
        color_arr = np.zeros((*cm.shape, 4))
        for i in range(n):
            for j in range(n):
                if i == j:
                    intensity = 0.3 + 0.6 * (cm_pct[i, j] / 100)
                    color_arr[i, j] = [0.18, 0.63, 0.27, intensity]  # green
                else:
                    intensity = 0.2 + 0.6 * (cm_pct[i, j] / 100) if cm_pct[i, j] > 0 else 0.05
                    color_arr[i, j] = [0.85, 0.15, 0.15, intensity]  # red

        ax.imshow(color_arr, aspect='auto')

        # Cell annotations: count on top line, % on second line
        for i in range(n):
            for j in range(n):
                text_color = 'white' if cm_pct[i, j] > 50 else 'black'
                label = f'{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)'
                ax.text(j, i, label, ha='center', va='center',
                        fontsize=12, fontweight='bold', color=text_color)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10, rotation=0)
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)

        # Grid lines between cells
        for x in np.arange(-0.5, n, 1):
            ax.axhline(x, color='white', linewidth=1.5)
            ax.axvline(x, color='white', linewidth=1.5)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save_name: str = None,
        normalize: bool = False
    ):
        """
        Plot a single confusion matrix with count + percentage in each cell.
        Correct predictions shown in green, errors in red.
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]

        fig, ax = plt.subplots(figsize=(6, 5))
        self._draw_confusion_matrix(ax, cm, class_names, title)
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.show()

    def plot_all_confusion_matrices(
        self,
        cms: Dict[str, np.ndarray],
        class_names: List[str] = None,
        title: str = "Confusion Matrices — All Models",
        save_name: str = None
    ):
        """
        Plot all model confusion matrices side-by-side in a 2×2 grid.

        Args:
            cms: {model_name: confusion_matrix_array}
            class_names: Class label names
            title: Overall figure title
            save_name: Filename to save
        """
        if class_names is None:
            class_names = ['Benign', 'Attack']

        names = list(cms.keys())
        n = len(names)
        ncols = 2
        nrows = (n + 1) // 2

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
        axes = np.array(axes).flatten()

        for idx, model_name in enumerate(names):
            self._draw_confusion_matrix(axes[idx], cms[model_name], class_names, model_name)

        # Hide any unused subplot slots
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"All confusion matrices saved to {save_path}")

        plt.show()
        
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve",
        save_name: str = None
    ):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title: Plot title
            save_name: Filename to save plot
        """
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})'
        )
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        
    def plot_feature_importance(
        self,
        feature_importance: List,
        feature_names: List[str] = None,
        top_n: int = 20,
        title: str = "Top Feature Importances",
        save_name: str = None
    ):
        """
        Plot feature importance
        
        Args:
            feature_importance: List of (feature_idx, importance) tuples
            feature_names: Names of features
            top_n: Number of top features to display
            title: Plot title
            save_name: Filename to save plot
        """
        # Extract top features
        top_features = feature_importance[:top_n]
        
        if feature_names is not None:
            labels = [feature_names[idx] for idx, _ in top_features]
        elif top_features and isinstance(top_features[0][0], str):
            labels = [name for name, _ in top_features]
        else:
            labels = [f"Feature {idx}" for idx, _ in top_features]
        
        importances = [imp for _, imp in top_features]
        
        # Plot
        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(labels))
        
        plt.barh(y_pos, importances, align='center', alpha=0.8, color='steelblue')
        plt.yticks(y_pos, labels)
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
    def plot_training_history(
        self,
        history: Dict,
        metrics: List[str] = ['loss', 'accuracy'],
        title: str = "Training History",
        save_name: str = None
    ):
        """
        Plot training history for neural networks
        
        Args:
            history: Training history dictionary
            metrics: Metrics to plot
            title: Plot title
            save_name: Filename to save plot
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Plot training metric
            if metric in history:
                ax.plot(history[metric], label=f'Train {metric}', linewidth=2)
            
            # Plot validation metric if available
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Val {metric}', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=11)
            ax.set_ylabel(metric.capitalize(), fontsize=11)
            ax.set_title(f'{metric.capitalize()} over Epochs', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
        
    def plot_model_comparison(
        self,
        results: Dict[str, Dict],
        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
        title: str = "Model Performance Comparison",
        save_name: str = None
    ):
        """
        Compare multiple models
        
        Args:
            results: Dictionary of {model_name: metrics_dict}
            metrics: Metrics to compare
            title: Plot title
            save_name: Filename to save plot
        """
        model_names = list(results.keys())
        n_models = len(model_names)
        n_metrics = len(metrics)

        # Distinct color per model
        model_colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800',
                        '#9C27B0', '#00BCD4', '#795548', '#607D8B']

        # Build data: rows = metrics, cols = models
        data = np.zeros((n_metrics, n_models))
        for mi, metric in enumerate(metrics):
            for mo, model_name in enumerate(model_names):
                data[mi, mo] = results[model_name].get(metric, 0)

        fig, ax = plt.subplots(figsize=(13, 7))

        # Metrics on x-axis, one bar-group per metric
        x = np.arange(n_metrics)
        width = 0.7 / n_models

        for mo, (model_name, color) in enumerate(zip(model_names, model_colors)):
            offset = (mo - n_models / 2) * width + width / 2
            bars = ax.bar(
                x + offset,
                data[:, mo],
                width,
                label=model_name,
                color=color,
                alpha=0.88,
                edgecolor='black',
                linewidth=0.5
            )
            # Value annotation on each bar
            for bar, val in zip(bars, data[:, mo]):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f'{val:.3f}',
                        ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold', rotation=90
                    )

        metric_labels = [m.replace('_', ' ').replace('roc auc', 'ROC-AUC').title()
                         for m in metrics]
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Zoom y-axis so differences are visible
        y_min = max(0, data[data > 0].min() - 0.08)
        ax.set_ylim([y_min, 1.06])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))

        ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")

        plt.show()
        
    def plot_roc_comparison(
        self,
        roc_data: dict,
        title: str = "Receiver Operating Characteristic (ROC) Curves",
        save_name: str = None
    ):
        """
        Plot ROC curves for multiple models on the same axes.

        Args:
            roc_data: {model_label: {'fpr': array, 'tpr': array, 'auc': float}}
            title: Plot title
            save_name: Filename to save plot
        """
        styles = [
            ('RF',  'blue',   '-',    2.0),
            ('NN',  'red',    '--',   2.0),
            ('SVM', 'green',  ':',    2.0),
            ('LR',  'orange', '-.',   2.0),
        ]
        style_map = {abbr: (c, ls, lw) for abbr, c, ls, lw in styles}

        fig, ax = plt.subplots(figsize=(8, 7))

        for label, data in roc_data.items():
            fpr, tpr, auc_val = data['fpr'], data['tpr'], data['auc']
            # Match abbreviated style key
            key = next((k for k in style_map if k in label.upper()), None)
            color, linestyle, lw = style_map.get(key, ('gray', '-', 1.5))
            ax.plot(fpr, tpr, color=color, linestyle=linestyle, linewidth=lw,
                    label=f'{label} (AUC={auc_val:.3f})')

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.2, label='Random Guess')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC comparison plot saved to {save_path}")

        plt.show()

    def plot_class_distribution(
        self,
        y: np.ndarray,
        class_names: List[str] = None,
        title: str = "Class Distribution",
        save_name: str = None
    ):
        """
        Plot class distribution
        
        Args:
            y: Labels
            class_names: Names of classes
            title: Plot title
            save_name: Filename to save plot
        """
        unique, counts = np.unique(y, return_counts=True)

        if class_names is None:
            class_names = [f'Class {i}' for i in unique]

        colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']
        bar_colors = [colors[i % len(colors)] for i in range(len(unique))]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1.01)

        # Bar chart
        ax = axes[0]
        bars = ax.bar(class_names, counts, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.7)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Sample Count', fontsize=12)
        ax.set_title('Sample Count per Class', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        total = counts.sum()
        for bar, count in zip(bars, counts):
            pct = 100 * count / total
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                bar.get_height() + total * 0.005,
                f'{int(count)}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold'
            )
        ax.set_ylim(0, counts.max() * 1.18)

        # Pie chart
        ax2 = axes[1]
        wedges, texts, autotexts = ax2.pie(
            counts,
            labels=class_names,
            colors=bar_colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            wedgeprops=dict(edgecolor='black', linewidth=0.7)
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_fontweight('bold')
        ax2.set_title('Class Proportion', fontsize=12, fontweight='bold')

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")

        plt.show()


if __name__ == "__main__":
    # Test visualizer
    logging.basicConfig(level=logging.INFO)
    
    viz = Visualizer()
    
    # Test confusion matrix
    cm = np.array([[85, 15], [10, 90]])
    viz.plot_confusion_matrix(cm, class_names=['Benign', 'Attack'])
    
    # Test ROC curve
    y_true = np.random.choice([0, 1], 200)
    y_proba = np.random.rand(200)
    viz.plot_roc_curve(y_true, y_proba)
