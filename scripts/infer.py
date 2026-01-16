"""
Run inference on new documents using a trained model.
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

from src.models import TransformerClassifier
from src.features import KeywordExtractor
from src.preprocessing import TextPreprocessor

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


def save_jsonl(documents: list, file_path: str):
    """Save documents to JSONL."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")


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
        description="Run inference on new documents"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to input documents JSONL"
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
        "--output_path",
        type=str,
        required=True,
        help="Path to save predictions JSONL"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--extract_keywords",
        action="store_true",
        help="Extract keywords for each document"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Running Inference")
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
    
    inference_config = config.get('inference', {})
    prep_config = config.get('preparation', {})
    
    # Step 1: Load documents
    logger.info(f"\n[1/5] Loading documents from {args.input_path}...")
    documents = load_jsonl(args.input_path)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Step 2: Preprocess
    logger.info("\n[2/5] Preprocessing documents...")
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
        else:
            # Keep original but mark as invalid
            doc['valid'] = False
            processed_docs.append(doc)
    
    logger.info(f"Valid documents: {sum(1 for d in processed_docs if d.get('valid', True))}")
    
    # Step 3: Load model and run inference
    logger.info(f"\n[3/5] Loading {args.model_type} model and running inference...")
    
    texts = [doc.get('text_clean', doc.get('text', '')) for doc in processed_docs]
    valid_mask = [doc.get('valid', True) for doc in processed_docs]
    
    if args.model_type == "baseline":
        model, vectorizer = load_baseline_model(args.model_path)
        X = vectorizer.transform(texts)
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
    else:
        model = TransformerClassifier.load(args.model_path)
        predictions = model.predict(texts, batch_size=args.batch_size)
        probabilities = model.predict_proba(texts, batch_size=args.batch_size)
    
    # Step 4: Extract keywords (optional)
    logger.info("\n[4/5] Processing predictions...")
    
    keyword_extractor = None
    if args.extract_keywords:
        logger.info("Extracting keywords...")
        phrases_config = config.get('phrases', {})
        keyword_extractor = KeywordExtractor(
            model_name=phrases_config.get('keybert_model', 'all-MiniLM-L6-v2'),
            top_n=phrases_config.get('top_n_keywords', 10),
            ngram_range=tuple(phrases_config.get('keyphrase_ngram_range', [1, 3])),
            diversity=phrases_config.get('diversity', 0.5),
        )
    
    # Add predictions to documents
    for i, doc in enumerate(processed_docs):
        if valid_mask[i]:
            pred_class = int(predictions[i])
            proba = probabilities[i]
            
            doc['predicted_class'] = pred_class
            doc['predicted_class_name'] = class_names[pred_class]
            doc['prob_left'] = float(proba[0])
            doc['prob_center'] = float(proba[1])
            doc['prob_right'] = float(proba[2])
            doc['polarity_score'] = float(proba[2] - proba[0])
            
            # Compute uncertainty (entropy)
            epsilon = 1e-10
            entropy = -np.sum(proba * np.log(proba + epsilon))
            doc['uncertainty'] = float(entropy)
            
            # Extract keywords
            if keyword_extractor:
                text = doc.get('text_clean', doc.get('text', ''))
                keywords = keyword_extractor.extract_keywords(text)
                doc['keywords'] = [kw for kw, _ in keywords]
            
        else:
            doc['predicted_class'] = None
            doc['predicted_class_name'] = None
            doc['error'] = "Document failed preprocessing"
    
    # Step 5: Save predictions
    logger.info(f"\n[5/5] Saving predictions to {args.output_path}...")
    save_jsonl(processed_docs, args.output_path)
    
    # Log statistics
    valid_preds = [doc for doc in processed_docs if doc.get('predicted_class') is not None]
    
    if valid_preds:
        class_counts = {}
        for doc in valid_preds:
            pred_class = doc['predicted_class']
            class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
        
        polarity_scores = [doc['polarity_score'] for doc in valid_preds]
        uncertainties = [doc['uncertainty'] for doc in valid_preds]
        
        logger.info("\n" + "=" * 60)
        logger.info("Inference Statistics")
        logger.info("=" * 60)
        logger.info(f"Total documents: {len(documents)}")
        logger.info(f"Valid predictions: {len(valid_preds)}")
        logger.info(f"Failed: {len(documents) - len(valid_preds)}")
        
        logger.info("\nPredicted class distribution:")
        for class_idx in sorted(class_counts.keys()):
            count = class_counts[class_idx]
            pct = 100 * count / len(valid_preds)
            logger.info(f"  {class_names[class_idx]}: {count} ({pct:.1f}%)")
        
        logger.info(f"\nPolarity scores:")
        logger.info(f"  Range: [{min(polarity_scores):.3f}, {max(polarity_scores):.3f}]")
        logger.info(f"  Mean: {np.mean(polarity_scores):.3f}")
        logger.info(f"  Median: {np.median(polarity_scores):.3f}")
        
        logger.info(f"\nUncertainty:")
        logger.info(f"  Mean: {np.mean(uncertainties):.3f}")
        logger.info(f"  Median: {np.median(uncertainties):.3f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ Inference complete!")
    logger.info(f"Predictions saved to {args.output_path}")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
