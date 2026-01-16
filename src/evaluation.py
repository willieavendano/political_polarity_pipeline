"""
Model evaluation metrics, calibration analysis, and error analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        logger.info("Computing evaluation metrics...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        macro_f1 = np.mean(f1)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            per_class_metrics[class_name] = {
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "support": int(support[i]),
            }
        
        # Overall metrics
        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": per_class_metrics,
            "confusion_matrix": cm.tolist(),
        }
        
        # Calibration metrics
        if y_proba is not None:
            ece = self.compute_expected_calibration_error(y_true, y_proba)
            metrics["expected_calibration_error"] = ece
        
        # Log summary
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {macro_f1:.4f}")
        for class_name, class_metrics in per_class_metrics.items():
            logger.info(
                f"{class_name}: P={class_metrics['precision']:.4f}, "
                f"R={class_metrics['recall']:.4f}, "
                f"F1={class_metrics['f1']:.4f}, "
                f"N={class_metrics['support']}"
            )
        
        return metrics
    
    def compute_expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            ECE value
        """
        # Get predicted class and confidence
        y_pred = np.argmax(y_proba, axis=1)
        confidences = np.max(y_proba, axis=1)
        accuracies = (y_pred == y_true).astype(float)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        output_path: str,
        normalize: bool = True,
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            output_path: Path to save plot
            normalize: Whether to normalize by true class
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
        )
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix saved to {output_path}")
    
    def plot_calibration_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        output_path: str,
        n_bins: int = 10,
    ):
        """
        Plot calibration curve for each class.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            output_path: Path to save plot
            n_bins: Number of bins
        """
        fig, axes = plt.subplots(1, self.num_classes, figsize=(15, 4))
        if self.num_classes == 1:
            axes = [axes]
        
        for i, (ax, class_name) in enumerate(zip(axes, self.class_names)):
            # Binary labels for this class
            y_true_binary = (y_true == i).astype(int)
            y_proba_class = y_proba[:, i]
            
            # Compute calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_binary,
                y_proba_class,
                n_bins=n_bins,
                strategy='uniform',
            )
            
            # Plot
            ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax.plot(
                mean_predicted_value,
                fraction_of_positives,
                'o-',
                label=class_name,
            )
            ax.set_xlabel('Mean predicted probability')
            ax.set_ylabel('Fraction of positives')
            ax.set_title(f'{class_name} Calibration')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Calibration curve saved to {output_path}")
    
    def plot_class_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
    ):
        """
        Plot true vs predicted class distributions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            output_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # True distribution
        true_counts = np.bincount(y_true, minlength=self.num_classes)
        axes[0].bar(range(self.num_classes), true_counts, color='steelblue')
        axes[0].set_xticks(range(self.num_classes))
        axes[0].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[0].set_ylabel('Count')
        axes[0].set_title('True Distribution')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Predicted distribution
        pred_counts = np.bincount(y_pred, minlength=self.num_classes)
        axes[1].bar(range(self.num_classes), pred_counts, color='coral')
        axes[1].set_xticks(range(self.num_classes))
        axes[1].set_xticklabels(self.class_names, rotation=45, ha='right')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Predicted Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        logger.info(f"Class distribution plot saved to {output_path}")
    
    def analyze_errors(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        doc_ids: Optional[List[str]] = None,
        max_samples: int = 100,
    ) -> pd.DataFrame:
        """
        Perform error analysis.
        
        Args:
            texts: Input texts
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            doc_ids: Optional document IDs
            max_samples: Maximum number of errors to return
            
        Returns:
            DataFrame with error analysis
        """
        logger.info("Performing error analysis...")
        
        # Find misclassified samples
        errors = y_true != y_pred
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            logger.info("No errors found!")
            return pd.DataFrame()
        
        # Sample if too many errors
        if len(error_indices) > max_samples:
            error_indices = np.random.choice(
                error_indices, max_samples, replace=False
            )
        
        # Compile error data
        error_data = []
        for idx in error_indices:
            true_class = self.class_names[y_true[idx]]
            pred_class = self.class_names[y_pred[idx]]
            confidence = y_proba[idx, y_pred[idx]]
            
            error_data.append({
                "doc_id": doc_ids[idx] if doc_ids else str(idx),
                "text_preview": texts[idx][:200] + "..." if len(texts[idx]) > 200 else texts[idx],
                "true_class": true_class,
                "predicted_class": pred_class,
                "confidence": confidence,
                "prob_left": y_proba[idx, 0],
                "prob_center": y_proba[idx, 1],
                "prob_right": y_proba[idx, 2],
            })
        
        df = pd.DataFrame(error_data)
        logger.info(f"Analyzed {len(df)} errors")
        
        return df
    
    def compute_polarity_scores(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Compute continuous polarity scores from probabilities.
        
        Score = P(right) - P(left), ranging from -1 (left) to +1 (right)
        
        Args:
            y_proba: Predicted probabilities [left, center, right]
            
        Returns:
            Array of polarity scores
        """
        # Assuming class order: 0=left, 1=center, 2=right
        scores = y_proba[:, 2] - y_proba[:, 0]
        return scores
    
    def compute_uncertainty(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Compute prediction uncertainty using entropy.
        
        Args:
            y_proba: Predicted probabilities
            
        Returns:
            Array of entropy values
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        entropy = -np.sum(y_proba * np.log(y_proba + epsilon), axis=1)
        return entropy


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
):
    """
    Save detailed classification report to file.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save report
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
    )
    
    with open(output_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
    
    logger.info(f"Classification report saved to {output_path}")
