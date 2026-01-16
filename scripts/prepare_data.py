"""
Prepare and split data for training.

Handles:
- Loading raw JSONL documents
- Applying preprocessing and deduplication
- Applying distant supervision (outlet labels)
- Optional manual labels
- Splitting into train/val/test sets
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    TextPreprocessor,
    Deduplicator,
    apply_distant_supervision,
    apply_manual_labels,
    stratified_split,
)

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


def load_outlet_labels(file_path: str) -> dict:
    """Load outlet labels from CSV."""
    df = pd.read_csv(file_path)
    return dict(zip(df['outlet_name'], df['bias_label']))


def load_manual_labels(file_path: str) -> dict:
    """Load manual labels from CSV."""
    df = pd.read_csv(file_path)
    return dict(zip(df['doc_id'], df['label']))


def save_jsonl(documents: list, file_path: str):
    """Save documents to JSONL."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data for political polarity classification"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--outlet_labels",
        type=str,
        default=None,
        help="Path to outlet labels CSV (distant supervision)"
    )
    parser.add_argument(
        "--manual_labels",
        type=str,
        default=None,
        help="Path to manual labels CSV (optional)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Test set proportion"
    )
    parser.add_argument(
        "--val_size",
        type=float,
        default=0.15,
        help="Validation set proportion"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Data Preparation Pipeline")
    logger.info("=" * 60)
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file {args.config} not found, using defaults")
        config = {}
    
    prep_config = config.get('preparation', {})
    label_config = config.get('labels', {})
    
    # Label mapping
    label_map = {
        'left': label_config.get('left', 0),
        'center': label_config.get('center', 1),
        'right': label_config.get('right', 2),
    }
    
    # Step 1: Load documents
    logger.info(f"\n[1/7] Loading documents from {args.input_path}...")
    documents = load_jsonl(args.input_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Preprocess
    logger.info("\n[2/7] Preprocessing documents...")
    preprocessor = TextPreprocessor(
        min_length=prep_config.get('min_text_length', 50),
        max_length=prep_config.get('max_text_length', 10000),
        language=prep_config.get('language', 'en'),
    )
    
    processed_docs = []
    for doc in documents:
        processed = preprocessor.process_document(doc)
        if processed is not None:
            processed_docs.append(processed)
    
    logger.info(
        f"After preprocessing: {len(processed_docs)}/{len(documents)} documents retained"
    )
    
    # Step 3: Deduplication
    logger.info("\n[3/7] Deduplicating documents...")
    deduplicator = Deduplicator()
    unique_docs = deduplicator.deduplicate_documents(processed_docs)
    
    # Step 4: Apply labels
    logger.info("\n[4/7] Applying labels...")
    
    labeled_docs = unique_docs
    
    # Distant supervision
    if args.outlet_labels:
        logger.info("Applying distant supervision (outlet labels)...")
        outlet_labels = load_outlet_labels(args.outlet_labels)
        labeled_docs = apply_distant_supervision(
            labeled_docs, outlet_labels, label_map
        )
    
    # Manual labels (override if provided)
    if args.manual_labels and os.path.exists(args.manual_labels):
        logger.info("Applying manual labels...")
        manual_labels = load_manual_labels(args.manual_labels)
        labeled_docs = apply_manual_labels(
            labeled_docs, manual_labels, override=True
        )
    
    # Filter to only labeled documents
    labeled_docs = [doc for doc in labeled_docs if 'label' in doc]
    logger.info(f"Total labeled documents: {len(labeled_docs)}")
    
    if len(labeled_docs) == 0:
        logger.error("No labeled documents! Check your label files.")
        return 1
    
    # Check label distribution
    label_counts = {}
    for doc in labeled_docs:
        label = doc['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("Label distribution:")
    for label, count in sorted(label_counts.items()):
        logger.info(f"  Class {label}: {count} documents")
    
    # Step 5: Split data
    logger.info("\n[5/7] Splitting data...")
    train_docs, val_docs, test_docs = stratified_split(
        labeled_docs,
        test_size=args.test_size,
        val_size=args.val_size,
        stratify_by_source=prep_config.get('stratify_by_source', True),
        random_state=args.random_seed,
    )
    
    # Step 6: Save processed data
    logger.info("\n[6/7] Saving processed data...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    save_jsonl(train_docs, train_path)
    save_jsonl(val_docs, val_path)
    save_jsonl(test_docs, test_path)
    
    logger.info(f"Saved {len(train_docs)} training documents to {train_path}")
    logger.info(f"Saved {len(val_docs)} validation documents to {val_path}")
    logger.info(f"Saved {len(test_docs)} test documents to {test_path}")
    
    # Step 7: Save metadata
    logger.info("\n[7/7] Saving metadata...")
    metadata = {
        "input_file": args.input_path,
        "total_documents": len(documents),
        "processed_documents": len(processed_docs),
        "unique_documents": len(unique_docs),
        "labeled_documents": len(labeled_docs),
        "train_size": len(train_docs),
        "val_size": len(val_docs),
        "test_size": len(test_docs),
        "label_distribution": label_counts,
        "random_seed": args.random_seed,
    }
    
    metadata_path = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved metadata to {metadata_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Data preparation complete!")
    logger.info(f"Next step: python -m scripts.train_baseline --train_path {train_path} --val_path {val_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
