"""
Train baseline model (TF-IDF + Logistic Regression).
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import TFIDFFeatureExtractor, PhraseExtractor
from src.models import BaselineClassifier
from src.evaluation import ModelEvaluator

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


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline classification model"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=True,
        help="Path to training data JSONL"
    )
    parser.add_argument(
        "--val_path",
        type=str,
        required=True,
        help="Path to validation data JSONL"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./artifacts/baseline",
        help="Output directory for model artifacts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Maximum TF-IDF features (overrides config)"
    )
    parser.add_argument(
        "--ngram_range",
        type=int,
        nargs=2,
        default=None,
        help="N-gram range (overrides config)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Training Baseline Model")
    logger.info("=" * 60)
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = {}
    
    baseline_config = config.get('baseline', {})
    label_config = config.get('labels', {})
    class_names = label_config.get('class_names', [
        "Left/Democrat/Liberal",
        "Center/Mixed/Unclear",
        "Right/Republican/Conservative",
    ])
    
    # Override config with CLI args if provided
    if args.max_features:
        baseline_config['max_features'] = args.max_features
    if args.ngram_range:
        baseline_config['ngram_range'] = args.ngram_range
    
    # Step 1: Load data
    logger.info(f"\n[1/6] Loading data...")
    train_docs = load_jsonl(args.train_path)
    val_docs = load_jsonl(args.val_path)
    
    logger.info(f"Training documents: {len(train_docs)}")
    logger.info(f"Validation documents: {len(val_docs)}")
    
    # Extract texts and labels
    train_texts = [doc['text_clean'] for doc in train_docs]
    train_labels = np.array([doc['label'] for doc in train_docs])
    
    val_texts = [doc['text_clean'] for doc in val_docs]
    val_labels = np.array([doc['label'] for doc in val_docs])
    
    # Step 2: Extract TF-IDF features
    logger.info("\n[2/6] Extracting TF-IDF features...")
    feature_extractor = TFIDFFeatureExtractor(
        max_features=baseline_config.get('max_features', 10000),
        ngram_range=tuple(baseline_config.get('ngram_range', [1, 3])),
        min_df=baseline_config.get('min_df', 5),
        max_df=baseline_config.get('max_df', 0.8),
        use_idf=baseline_config.get('use_idf', True),
        sublinear_tf=baseline_config.get('sublinear_tf', True),
    )
    
    X_train = feature_extractor.fit_transform(train_texts)
    X_val = feature_extractor.transform(val_texts)
    feature_names = feature_extractor.get_feature_names()
    
    logger.info(f"Feature matrix shape: {X_train.shape}")
    
    # Step 3: Train classifier
    logger.info("\n[3/6] Training classifier...")
    classifier = BaselineClassifier(
        classifier_type=baseline_config.get('classifier', 'logistic'),
        C=baseline_config.get('C', 1.0),
        class_weight=baseline_config.get('class_weight', 'balanced'),
        calibration=baseline_config.get('calibration', 'isotonic'),
        random_state=config.get('random_seed', 42),
    )
    
    classifier.train(X_train, train_labels, X_val, val_labels)
    
    # Step 4: Extract interpretable phrases
    logger.info("\n[4/6] Extracting discriminative phrases...")
    phrase_extractor = PhraseExtractor(
        top_n=baseline_config.get('top_features', 30)
    )
    class_phrases = phrase_extractor.extract_chi2_phrases(
        X_train, train_labels, feature_names, num_classes=len(class_names)
    )
    
    # Also get feature importance from model
    feature_importance = classifier.get_feature_importance(
        feature_names,
        top_n=baseline_config.get('top_features', 30)
    )
    
    # Log top phrases per class
    for class_idx, class_name in enumerate(class_names):
        logger.info(f"\nTop phrases for {class_name}:")
        if class_idx in feature_importance:
            top_phrases = feature_importance[class_idx][:10]
            for phrase, score in top_phrases:
                logger.info(f"  {phrase}: {score:.4f}")
    
    # Step 5: Evaluate on validation set
    logger.info("\n[5/6] Evaluating on validation set...")
    evaluator = ModelEvaluator(class_names)
    
    val_pred = classifier.predict(X_val)
    val_proba = classifier.predict_proba(X_val)
    
    metrics = evaluator.evaluate(val_labels, val_pred, val_proba)
    
    # Step 6: Save model and artifacts
    logger.info("\n[6/6] Saving model and artifacts...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(classifier, f)
    logger.info(f"Saved model to {model_path}")
    
    # Save feature extractor
    vectorizer_path = os.path.join(args.output_dir, "vectorizer.pkl")
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(feature_extractor, f)
    logger.info(f"Saved vectorizer to {vectorizer_path}")
    
    # Save feature importance
    importance_path = os.path.join(args.output_dir, "feature_importance.json")
    importance_data = {
        str(class_idx): [
            {"phrase": phrase, "score": float(score)}
            for phrase, score in features
        ]
        for class_idx, features in feature_importance.items()
    }
    with open(importance_path, 'w') as f:
        json.dump(importance_data, f, indent=2)
    logger.info(f"Saved feature importance to {importance_path}")
    
    # Save discriminative phrases
    phrases_path = os.path.join(args.output_dir, "discriminative_phrases.json")
    phrases_data = {
        str(class_idx): [
            {"phrase": phrase, "score": float(score)}
            for phrase, score in phrases
        ]
        for class_idx, phrases in class_phrases.items()
    }
    with open(phrases_path, 'w') as f:
        json.dump(phrases_data, f, indent=2)
    logger.info(f"Saved discriminative phrases to {phrases_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "val_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved validation metrics to {metrics_path}")
    
    # Save metadata
    metadata = {
        "model_type": "baseline",
        "classifier": baseline_config.get('classifier', 'logistic'),
        "max_features": baseline_config.get('max_features', 10000),
        "ngram_range": baseline_config.get('ngram_range', [1, 3]),
        "num_features": len(feature_names),
        "train_size": len(train_docs),
        "val_size": len(val_docs),
        "class_names": class_names,
        "val_metrics": metrics,
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Baseline model training complete!")
    logger.info(f"Validation accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Next step: python -m scripts.evaluate --test_path ./data/processed/test.jsonl --model_path {args.output_dir} --model_type baseline")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
