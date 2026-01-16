"""
Evaluate trained models on test data.

Produces comprehensive evaluation reports including:
- Metrics (accuracy, F1, precision, recall)
- Confusion matrix
- Calibration analysis
- Error analysis
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TransformerClassifier
from src.evaluation import ModelEvaluator, save_classification_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> list:
    """Load JSONL file."""
    documents = []
    with open(file_path, 'r') as f:
        for line in f:
            doc = json.loads(line.strip())
            documents.append(doc)
    return documents


def load_baseline_model(model_path: str):
    """Load baseline model and vectorizer."""
    model_file = os.path.join(model_path, "model.pkl")
    vectorizer_file = os.path.join(model_path, "vectorizer.pkl")
    
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    
    with open(vectorizer_file, 'rb') as f:
        vectorizer = pickle.load(f)
    
    return model, vectorizer


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test data"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=True,
        help="Path to test data JSONL"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "transformer"],
        help="Model type"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for evaluation results (default: model_path/evaluation)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, "evaluation")
    
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    label_config = config.get('labels', {})
    class_names = label_config.get('class_names', [
        "Left/Democrat/Liberal",
        "Center/Mixed/Unclear",
        "Right/Republican/Conservative",
    ])
    
    # Step 1: Load test data
    logger.info(f"\n[1/5] Loading test data...")
    test_docs = load_jsonl(args.test_path)
    logger.info(f"Test documents: {len(test_docs)}")
    
    test_texts = [doc['text_clean'] for doc in test_docs]
    test_labels = np.array([doc['label'] for doc in test_docs])
    test_ids = [doc.get('doc_id', str(i)) for i, doc in enumerate(test_docs)]
    
    # Step 2: Load model
    logger.info(f"\n[2/5] Loading {args.model_type} model...")
    
    if args.model_type == "baseline":
        model, vectorizer = load_baseline_model(args.model_path)
        X_test = vectorizer.transform(test_texts)
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)
    else:
        model = TransformerClassifier.load(args.model_path)
        test_pred = model.predict(test_texts)
        test_proba = model.predict_proba(test_texts)
    
    logger.info("Model loaded successfully")
    
    # Step 3: Evaluate
    logger.info(f"\n[3/5] Computing evaluation metrics...")
    evaluator = ModelEvaluator(class_names)
    
    metrics = evaluator.evaluate(test_labels, test_pred, test_proba)
    
    # Step 4: Generate visualizations
    logger.info(f"\n[4/5] Generating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    evaluator.plot_confusion_matrix(
        np.array(metrics['confusion_matrix']),
        cm_path,
        normalize=True,
    )
    
    # Calibration curve
    calib_path = os.path.join(args.output_dir, "calibration_curve.png")
    evaluator.plot_calibration_curve(
        test_labels,
        test_proba,
        calib_path,
    )
    
    # Class distribution
    dist_path = os.path.join(args.output_dir, "class_distribution.png")
    evaluator.plot_class_distribution(
        test_labels,
        test_pred,
        dist_path,
    )
    
    # Step 5: Error analysis
    logger.info(f"\n[5/5] Performing error analysis...")
    error_df = evaluator.analyze_errors(
        test_texts,
        test_labels,
        test_pred,
        test_proba,
        doc_ids=test_ids,
        max_samples=100,
    )
    
    if not error_df.empty:
        error_path = os.path.join(args.output_dir, "error_analysis.csv")
        error_df.to_csv(error_path, index=False)
        logger.info(f"Saved error analysis to {error_path}")
    
    # Save detailed classification report
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    save_classification_report(
        test_labels,
        test_pred,
        class_names,
        report_path,
    )
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved test metrics to {metrics_path}")
    
    # Compute and save polarity scores
    polarity_scores = evaluator.compute_polarity_scores(test_proba)
    uncertainty = evaluator.compute_uncertainty(test_proba)
    
    results_df = pd.DataFrame({
        'doc_id': test_ids,
        'true_label': test_labels,
        'predicted_label': test_pred,
        'prob_left': test_proba[:, 0],
        'prob_center': test_proba[:, 1],
        'prob_right': test_proba[:, 2],
        'polarity_score': polarity_scores,
        'uncertainty': uncertainty,
    })
    
    results_path = os.path.join(args.output_dir, "predictions.csv")
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved predictions to {results_path}")
    
    # Log summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results Summary")
    logger.info("=" * 60)
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Macro-F1: {metrics['macro_f1']:.4f}")
    if 'expected_calibration_error' in metrics:
        logger.info(f"Expected Calibration Error: {metrics['expected_calibration_error']:.4f}")
    
    logger.info("\nPer-class metrics:")
    for class_name, class_metrics in metrics['per_class'].items():
        logger.info(
            f"  {class_name}:"
            f" P={class_metrics['precision']:.4f},"
            f" R={class_metrics['recall']:.4f},"
            f" F1={class_metrics['f1']:.4f}"
        )
    
    logger.info(f"\nPolarity score range: [{polarity_scores.min():.3f}, {polarity_scores.max():.3f}]")
    logger.info(f"Mean polarity score: {polarity_scores.mean():.3f}")
    logger.info(f"Mean uncertainty: {uncertainty.mean():.3f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Evaluation complete!")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
