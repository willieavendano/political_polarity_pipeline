"""
Feature extraction and phrase/keyword identification.

Implements TF-IDF, n-grams, statistical phrase extraction, and keyword extraction.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class TFIDFFeatureExtractor:
    """Extracts TF-IDF features from text."""
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 5,
        max_df: float = 0.8,
        use_idf: bool = True,
        sublinear_tf: bool = True,
    ):
        """
        Initialize TF-IDF extractor.
        
        Args:
            max_features: Maximum number of features
            ngram_range: Range of n-grams to extract
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF weighting
            sublinear_tf: Whether to use sublinear TF scaling
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            use_idf=use_idf,
            sublinear_tf=sublinear_tf,
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
        )
        self.feature_names: List[str] = []
    
    def fit(self, texts: List[str]) -> "TFIDFFeatureExtractor":
        """
        Fit vectorizer on texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Self
        """
        logger.info(f"Fitting TF-IDF on {len(texts)} documents...")
        self.vectorizer.fit(texts)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        logger.info(f"Extracted {len(self.feature_names)} features")
        return self
    
    def transform(self, texts: List[str]) -> csr_matrix:
        """
        Transform texts to TF-IDF features.
        
        Args:
            texts: List of text strings
            
        Returns:
            Sparse matrix of TF-IDF features
        """
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts: List[str]) -> csr_matrix:
        """Fit and transform in one step."""
        logger.info(f"Fitting and transforming TF-IDF on {len(texts)} documents...")
        features = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out().tolist()
        logger.info(f"Extracted {len(self.feature_names)} features")
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names (n-grams)."""
        return self.feature_names


class PhraseExtractor:
    """Extracts discriminative phrases using statistical methods."""
    
    def __init__(self, top_n: int = 30):
        """
        Initialize phrase extractor.
        
        Args:
            top_n: Number of top phrases to extract per class
        """
        self.top_n = top_n
        self.class_phrases: Dict[int, List[Tuple[str, float]]] = {}
    
    def extract_chi2_phrases(
        self,
        features: csr_matrix,
        labels: np.ndarray,
        feature_names: List[str],
        num_classes: int = 3,
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract discriminative phrases using chi-squared test.
        
        Args:
            features: TF-IDF feature matrix
            labels: Class labels
            feature_names: List of feature names
            num_classes: Number of classes
            
        Returns:
            Dictionary mapping class to list of (phrase, score) tuples
        """
        logger.info("Extracting discriminative phrases using chi-squared...")
        
        class_phrases = {}
        
        for class_idx in range(num_classes):
            # Create binary labels for this class
            binary_labels = (labels == class_idx).astype(int)
            
            # Compute chi-squared scores
            chi2_scores, p_values = chi2(features, binary_labels)
            
            # Get top features for this class
            top_indices = np.argsort(chi2_scores)[::-1][:self.top_n]
            top_phrases = [
                (feature_names[idx], chi2_scores[idx])
                for idx in top_indices
                if not np.isnan(chi2_scores[idx])
            ]
            
            class_phrases[class_idx] = top_phrases
            logger.info(f"Class {class_idx}: extracted {len(top_phrases)} phrases")
        
        self.class_phrases = class_phrases
        return class_phrases
    
    def get_top_phrases(self, class_idx: int, top_n: Optional[int] = None) -> List[str]:
        """
        Get top phrases for a class.
        
        Args:
            class_idx: Class index
            top_n: Number of phrases to return (default: self.top_n)
            
        Returns:
            List of phrase strings
        """
        if class_idx not in self.class_phrases:
            return []
        
        n = top_n if top_n is not None else self.top_n
        return [phrase for phrase, _ in self.class_phrases[class_idx][:n]]


class KeywordExtractor:
    """Extracts keywords using embedding-based methods."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_n: int = 20,
        ngram_range: Tuple[int, int] = (1, 3),
        diversity: float = 0.5,
    ):
        """
        Initialize keyword extractor.
        
        Args:
            model_name: Sentence transformer model name
            top_n: Number of keywords to extract
            ngram_range: Range of n-grams for keywords
            diversity: Diversity parameter for MMR (0=similarity, 1=diversity)
        """
        self.model_name = model_name
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.diversity = diversity
        self.model = None
    
    def _load_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                from keybert import KeyBERT
                self.model = KeyBERT(model=self.model_name)
                logger.info(f"Loaded KeyBERT model: {self.model_name}")
            except ImportError:
                logger.warning(
                    "KeyBERT not available. Install with: pip install keybert"
                )
                self.model = False
    
    def extract_keywords(
        self,
        text: str,
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        self._load_model()
        
        if self.model is False:
            # Fallback: simple TF-IDF based extraction
            return self._fallback_extraction(text, top_n)
        
        n = top_n if top_n is not None else self.top_n
        
        try:
            keywords = self.model.extract_keywords(
                text,
                keyphrase_ngram_range=self.ngram_range,
                stop_words="english",
                top_n=n,
                use_mmr=True,
                diversity=self.diversity,
            )
            return keywords
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}. Using fallback.")
            return self._fallback_extraction(text, top_n)
    
    def _fallback_extraction(
        self,
        text: str,
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Fallback keyword extraction using simple TF-IDF."""
        n = top_n if top_n is not None else self.top_n
        
        # Simple TF-IDF on single document
        vectorizer = TfidfVectorizer(
            max_features=n * 2,
            ngram_range=self.ngram_range,
            stop_words="english",
        )
        
        try:
            tfidf = vectorizer.fit_transform([text])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf.toarray()[0]
            
            # Sort by score
            top_indices = np.argsort(scores)[::-1][:n]
            keywords = [
                (feature_names[idx], scores[idx])
                for idx in top_indices
                if scores[idx] > 0
            ]
            return keywords
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")
            return []
    
    def extract_keywords_batch(
        self,
        texts: List[str],
        top_n: Optional[int] = None,
    ) -> List[List[Tuple[str, float]]]:
        """
        Extract keywords from multiple texts.
        
        Args:
            texts: List of input texts
            top_n: Number of keywords per text
            
        Returns:
            List of keyword lists
        """
        return [self.extract_keywords(text, top_n) for text in texts]


def compute_class_tfidf(
    texts_by_class: Dict[int, List[str]],
    max_features: int = 100,
    ngram_range: Tuple[int, int] = (1, 3),
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Compute class-specific TF-IDF to find characteristic phrases.
    
    This treats all documents in a class as one large pseudo-document.
    
    Args:
        texts_by_class: Dictionary mapping class to list of texts
        max_features: Maximum features to extract per class
        ngram_range: N-gram range
        
    Returns:
        Dictionary mapping class to list of (phrase, score) tuples
    """
    logger.info("Computing class-specific TF-IDF...")
    
    class_phrases = {}
    
    # Combine all texts per class
    class_texts = {
        cls: " ".join(texts) for cls, texts in texts_by_class.items()
    }
    
    # Create corpus for TF-IDF (all classes)
    corpus = list(class_texts.values())
    classes = list(class_texts.keys())
    
    # Fit TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features * len(classes),
        ngram_range=ngram_range,
        stop_words="english",
    )
    tfidf = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()
    
    # Extract top features per class
    for idx, cls in enumerate(classes):
        scores = tfidf[idx].toarray()[0]
        top_indices = np.argsort(scores)[::-1][:max_features]
        top_phrases = [
            (feature_names[i], scores[i])
            for i in top_indices
            if scores[i] > 0
        ]
        class_phrases[cls] = top_phrases
        logger.info(f"Class {cls}: {len(top_phrases)} characteristic phrases")
    
    return class_phrases
