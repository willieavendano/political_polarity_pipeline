"""
Tests for feature extraction module.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from src.features import TFIDFFeatureExtractor, PhraseExtractor, compute_class_tfidf


class TestTFIDFFeatureExtractor:
    """Tests for TFIDFFeatureExtractor class."""
    
    def test_fit_transform(self):
        """Test TF-IDF fitting and transformation."""
        texts = [
            "This is a test document about politics",
            "Another document discussing political issues",
            "A third document on different topics",
        ]
        
        extractor = TFIDFFeatureExtractor(max_features=100, ngram_range=(1, 2))
        features = extractor.fit_transform(texts)
        
        # Check output shape
        assert features.shape[0] == len(texts)
        assert features.shape[1] <= 100
        assert isinstance(features, csr_matrix)
        
        # Check feature names
        feature_names = extractor.get_feature_names()
        assert len(feature_names) > 0
        assert isinstance(feature_names[0], str)
    
    def test_transform(self):
        """Test transformation on new data."""
        train_texts = [
            "Training document one",
            "Training document two",
        ]
        
        test_texts = [
            "Test document",
        ]
        
        extractor = TFIDFFeatureExtractor()
        extractor.fit(train_texts)
        
        train_features = extractor.transform(train_texts)
        test_features = extractor.transform(test_texts)
        
        # Same number of features
        assert train_features.shape[1] == test_features.shape[1]


class TestPhraseExtractor:
    """Tests for PhraseExtractor class."""
    
    def test_extract_chi2_phrases(self):
        """Test chi-squared phrase extraction."""
        # Create simple TF-IDF features
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        texts = [
            "left liberal progressive democrat",
            "left progressive policy reform",
            "right conservative republican policy",
            "right republican conservative tradition",
        ]
        
        labels = np.array([0, 0, 1, 1])  # Binary for simplicity
        
        vectorizer = TfidfVectorizer(max_features=20)
        features = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        extractor = PhraseExtractor(top_n=5)
        class_phrases = extractor.extract_chi2_phrases(
            features, labels, feature_names, num_classes=2
        )
        
        # Check output
        assert 0 in class_phrases
        assert 1 in class_phrases
        assert len(class_phrases[0]) <= 5
        assert len(class_phrases[1]) <= 5
        
        # Check format
        for phrase, score in class_phrases[0]:
            assert isinstance(phrase, str)
            assert isinstance(score, (int, float))
    
    def test_get_top_phrases(self):
        """Test getting top phrases for a class."""
        extractor = PhraseExtractor(top_n=10)
        
        # Manually set phrases
        extractor.class_phrases = {
            0: [("phrase1", 0.9), ("phrase2", 0.8)],
            1: [("phrase3", 0.7)],
        }
        
        phrases_0 = extractor.get_top_phrases(0)
        assert len(phrases_0) == 2
        assert "phrase1" in phrases_0
        
        phrases_1 = extractor.get_top_phrases(1, top_n=1)
        assert len(phrases_1) == 1


class TestComputeClassTFIDF:
    """Tests for class-specific TF-IDF."""
    
    def test_compute_class_tfidf(self):
        """Test class-specific TF-IDF computation."""
        texts_by_class = {
            0: ["left wing progressive policies", "liberal democratic reforms"],
            1: ["centrist moderate balanced approach", "middle ground pragmatic"],
            2: ["right wing conservative values", "republican traditional policies"],
        }
        
        class_phrases = compute_class_tfidf(
            texts_by_class,
            max_features=10,
            ngram_range=(1, 2),
        )
        
        # Check output
        assert len(class_phrases) == 3
        for class_idx in range(3):
            assert class_idx in class_phrases
            assert len(class_phrases[class_idx]) > 0
            
            # Check format
            for phrase, score in class_phrases[class_idx]:
                assert isinstance(phrase, str)
                assert isinstance(score, (int, float))
                assert score >= 0
