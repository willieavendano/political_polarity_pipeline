"""
Interpretability tools for understanding model predictions.

Provides feature importance, attention analysis, and keyword-based explanations.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class BaselineInterpreter:
    """Interpretability for baseline models (TF-IDF + linear classifiers)."""
    
    def __init__(self, model, feature_names: List[str]):
        """
        Initialize interpreter.
        
        Args:
            model: Trained baseline model
            feature_names: List of feature names (n-grams)
        """
        self.model = model
        self.feature_names = feature_names
        self.coefficients = self._extract_coefficients()
    
    def _extract_coefficients(self) -> Optional[np.ndarray]:
        """Extract model coefficients."""
        # Handle calibrated models
        if hasattr(self.model, "calibrated_classifiers_"):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "coef_"):
                return base_estimator.coef_
        elif hasattr(self.model, "coef_"):
            return self.model.coef_
        
        logger.warning("Could not extract coefficients from model")
        return None
    
    def get_top_features_per_class(
        self,
        top_n: int = 30,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top features contributing to each class.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary mapping class index to list of (feature, weight) tuples
        """
        if self.coefficients is None:
            return {}
        
        class_features = {}
        
        for class_idx in range(self.coefficients.shape[0]):
            coef = self.coefficients[class_idx]
            
            # Get top positive features
            top_pos_indices = np.argsort(coef)[::-1][:top_n]
            top_features = [
                (self.feature_names[idx], coef[idx])
                for idx in top_pos_indices
            ]
            
            class_features[class_idx] = top_features
        
        return class_features
    
    def explain_prediction(
        self,
        text_features: csr_matrix,
        top_n: int = 10,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Explain a single prediction by showing contributing features.
        
        Args:
            text_features: TF-IDF features for a single document
            top_n: Number of top contributing features
            
        Returns:
            Dictionary mapping class to contributing features
        """
        if self.coefficients is None:
            return {}
        
        # Ensure text_features is 2D
        if text_features.shape[0] != 1:
            raise ValueError("Expected features for a single document")
        
        # Compute contribution scores
        contributions = {}
        feature_values = text_features.toarray()[0]
        
        for class_idx in range(self.coefficients.shape[0]):
            coef = self.coefficients[class_idx]
            
            # Element-wise multiplication
            scores = feature_values * coef
            
            # Get top contributing features
            top_indices = np.argsort(np.abs(scores))[::-1][:top_n]
            top_contrib = [
                (self.feature_names[idx], scores[idx])
                for idx in top_indices
                if scores[idx] != 0
            ]
            
            contributions[class_idx] = top_contrib
        
        return contributions


class TransformerInterpreter:
    """Interpretability for transformer models."""
    
    def __init__(self, model, tokenizer):
        """
        Initialize interpreter.
        
        Args:
            model: Trained transformer model
            tokenizer: Model tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
    
    def get_attention_weights(
        self,
        text: str,
        max_length: int = 256,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Get attention weights for a text.
        
        Args:
            text: Input text
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (tokens, attention_weights)
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        # Forward pass with attention output
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Aggregate attention across layers and heads
        # Shape: (num_layers, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        
        # Average across layers and heads, take attention to CLS token
        # This is a simple heuristic
        avg_attention = torch.stack(attentions).mean(dim=(0, 1))  # (seq_len, seq_len)
        cls_attention = avg_attention[0].cpu().numpy()  # Attention from CLS token
        
        return tokens, cls_attention
    
    def explain_prediction(
        self,
        text: str,
        top_n: int = 10,
        max_length: int = 256,
    ) -> List[Tuple[str, float]]:
        """
        Explain prediction using attention weights.
        
        Args:
            text: Input text
            top_n: Number of top tokens to return
            max_length: Maximum sequence length
            
        Returns:
            List of (token, attention_score) tuples
        """
        tokens, attention = self.get_attention_weights(text, max_length)
        
        # Filter out special tokens
        token_scores = [
            (token, score)
            for token, score in zip(tokens, attention)
            if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']
        ]
        
        # Sort by attention score
        token_scores.sort(key=lambda x: x[1], reverse=True)
        
        return token_scores[:top_n]
    
    def leave_one_out_analysis(
        self,
        text: str,
        keywords: List[str],
        max_length: int = 256,
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze impact of removing each keyword.
        
        Args:
            text: Input text
            keywords: List of keywords to test
            max_length: Maximum sequence length
            
        Returns:
            Dictionary mapping keyword to probability changes
        """
        self.model.eval()
        
        # Get original prediction
        original_proba = self._predict_text(text, max_length)
        
        results = {}
        
        for keyword in keywords:
            # Create text with keyword removed
            modified_text = text.replace(keyword, "")
            
            # Get new prediction
            modified_proba = self._predict_text(modified_text, max_length)
            
            # Compute changes
            changes = {
                "left_change": float(modified_proba[0] - original_proba[0]),
                "center_change": float(modified_proba[1] - original_proba[1]),
                "right_change": float(modified_proba[2] - original_proba[2]),
            }
            
            results[keyword] = changes
        
        return results
    
    def _predict_text(self, text: str, max_length: int) -> np.ndarray:
        """Helper to predict single text."""
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits
            proba = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        return proba


def extract_prediction_keywords(
    text: str,
    explanation: List[Tuple[str, float]],
    min_score: float = 0.01,
) -> List[str]:
    """
    Extract meaningful keywords from explanation.
    
    Args:
        text: Original text
        explanation: List of (token, score) tuples
        min_score: Minimum score threshold
        
    Returns:
        List of keyword strings
    """
    # Filter by score
    filtered = [
        token for token, score in explanation
        if score >= min_score
    ]
    
    # Reconstruct words from subword tokens
    keywords = []
    current_word = ""
    
    for token in filtered:
        if token.startswith("##"):
            # Continuation of previous word (BERT-style)
            current_word += token[2:]
        elif token.startswith("Ġ"):
            # New word (GPT-style)
            if current_word:
                keywords.append(current_word)
            current_word = token[1:]
        else:
            if current_word:
                keywords.append(current_word)
            current_word = token
    
    if current_word:
        keywords.append(current_word)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique_keywords.append(kw)
    
    return unique_keywords
