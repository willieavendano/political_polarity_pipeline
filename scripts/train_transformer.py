"""
Train transformer model (DistilBERT).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import TransformerClassifier
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
        description="Train transformer classification model"
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
        default="./artifacts/transformer",
        help="Output directory for model artifacts"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name (overrides config)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Training Transformer Model")
    logger.info("=" * 60)
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = {}
    
    transformer_config = config.get('transformer', {})
    label_config = config.get('labels', {})
    class_names = label_config.get('class_names', [
        "Left/Democrat/Liberal",
        "Center/Mixed/Unclear",
        "Right/Republican/Conservative",
    ])
    
    # Override config with CLI args if provided
    if args.model_name:
        transformer_config['model_name'] = args.model_name
    if args.epochs:
        transformer_config['epochs'] = args.epochs
    if args.batch_size:
        transformer_config['batch_size'] = args.batch_size
    if args.learning_rate:
        transformer_config['learning_rate'] = args.learning_rate
    
    # Step 1: Load data
    logger.info(f"\n[1/5] Loading data...")
    train_docs = load_jsonl(args.train_path)
    val_docs = load_jsonl(args.val_path)
    
    logger.info(f"Training documents: {len(train_docs)}")
    logger.info(f"Validation documents: {len(val_docs)}")
    
    # Extract texts and labels
    train_texts = [doc['text_clean'] for doc in train_docs]
    train_labels = [doc['label'] for doc in train_docs]
    
    val_texts = [doc['text_clean'] for doc in val_docs]
    val_labels = [doc['label'] for doc in val_docs]
    
    # Step 2: Initialize model
    logger.info("\n[2/5] Initializing transformer model...")
    model_name = transformer_config.get('model_name', 'distilbert-base-uncased')
    logger.info(f"Model: {model_name}")
    
    classifier = TransformerClassifier(
        model_name=model_name,
        num_classes=len(class_names),
        max_length=transformer_config.get('max_length', 256),
    )
    
    # Step 3: Train
    logger.info("\n[3/5] Training model...")
    logger.info("This may take a while depending on dataset size and hardware...")
    
    classifier.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        output_dir=args.output_dir,
        epochs=transformer_config.get('epochs', 3),
        batch_size=transformer_config.get('batch_size', 16),
        learning_rate=transformer_config.get('learning_rate', 2e-5),
        weight_decay=transformer_config.get('weight_decay', 0.01),
        warmup_steps=transformer_config.get('warmup_steps', 500),
        logging_steps=transformer_config.get('logging_steps', 100),
        eval_steps=transformer_config.get('eval_steps', 500),
        save_steps=transformer_config.get('save_steps', 500),
        early_stopping_patience=transformer_config.get('early_stopping_patience', 3),
        fp16=transformer_config.get('fp16', True),
    )
    
    # Step 4: Evaluate on validation set
    logger.info("\n[4/5] Evaluating on validation set...")
    evaluator = ModelEvaluator(class_names)
    
    val_pred = classifier.predict(val_texts)
    val_proba = classifier.predict_proba(val_texts)
    val_labels_array = np.array(val_labels)
    
    metrics = evaluator.evaluate(val_labels_array, val_pred, val_proba)
    
    # Step 5: Save final model and metadata
    logger.info("\n[5/5] Saving final model and metadata...")
    
    # Save model (already saved by trainer, but save again to be sure)
    classifier.save(args.output_dir)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, "val_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved validation metrics to {metrics_path}")
    
    # Save metadata
    metadata = {
        "model_type": "transformer",
        "model_name": model_name,
        "max_length": transformer_config.get('max_length', 256),
        "train_size": len(train_docs),
        "val_size": len(val_docs),
        "class_names": class_names,
        "epochs": transformer_config.get('epochs', 3),
        "batch_size": transformer_config.get('batch_size', 16),
        "learning_rate": transformer_config.get('learning_rate', 2e-5),
        "val_metrics": metrics,
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Transformer model training complete!")
    logger.info(f"Validation accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Validation macro-F1: {metrics['macro_f1']:.4f}")
    logger.info(f"Next step: python -m scripts.evaluate --test_path ./data/processed/test.jsonl --model_path {args.output_dir} --model_type transformer")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
