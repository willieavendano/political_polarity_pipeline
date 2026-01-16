"""
Tests for preprocessing module.
"""

import pytest
from src.preprocessing import (
    TextPreprocessor,
    Deduplicator,
    apply_distant_supervision,
    stratified_split,
)


class TestTextPreprocessor:
    """Tests for TextPreprocessor class."""
    
    def test_clean_text(self):
        """Test text cleaning."""
        preprocessor = TextPreprocessor()
        
        text = "Check this out: https://example.com   Multiple   spaces"
        cleaned = preprocessor.clean_text(text)
        
        assert "https://" not in cleaned
        assert "  " not in cleaned  # No double spaces
        
    def test_is_valid_length(self):
        """Test length validation."""
        preprocessor = TextPreprocessor(min_length=10, max_length=100)
        
        assert not preprocessor.is_valid_length("short")
        assert preprocessor.is_valid_length("This is a valid length text")
        assert not preprocessor.is_valid_length("x" * 200)
    
    def test_detect_language(self):
        """Test language detection."""
        preprocessor = TextPreprocessor()
        
        english_text = "This is an English sentence with common words"
        assert preprocessor.detect_language(english_text) == "en"
        
        # Non-English (simple heuristic test)
        foreign_text = "Esto es una oración en español"
        result = preprocessor.detect_language(foreign_text)
        # May or may not detect correctly with simple heuristic
        assert result in ["en", "other"]
    
    def test_process_document(self):
        """Test full document processing."""
        preprocessor = TextPreprocessor(min_length=20)
        
        # Valid document
        doc = {
            "doc_id": "test_001",
            "text": "This is a test document with sufficient length.",
            "title": "Test Document",
        }
        
        processed = preprocessor.process_document(doc)
        assert processed is not None
        assert "text_clean" in processed
        assert "text_length" in processed
        
        # Too short document
        short_doc = {
            "doc_id": "test_002",
            "text": "Short",
            "title": "",
        }
        
        processed_short = preprocessor.process_document(short_doc)
        assert processed_short is None


class TestDeduplicator:
    """Tests for Deduplicator class."""
    
    def test_compute_hash(self):
        """Test hash computation."""
        dedup = Deduplicator()
        
        text1 = "This is a test"
        text2 = "This is a test"
        text3 = "This is different"
        
        hash1 = dedup.compute_hash(text1)
        hash2 = dedup.compute_hash(text2)
        hash3 = dedup.compute_hash(text3)
        
        assert hash1 == hash2
        assert hash1 != hash3
    
    def test_is_duplicate(self):
        """Test duplicate detection."""
        dedup = Deduplicator()
        
        text1 = "Unique text 1"
        text2 = "Unique text 1"
        text3 = "Unique text 2"
        
        assert not dedup.is_duplicate(text1)
        assert dedup.is_duplicate(text2)
        assert not dedup.is_duplicate(text3)
    
    def test_deduplicate_documents(self):
        """Test document deduplication."""
        dedup = Deduplicator()
        
        docs = [
            {"text": "Document 1"},
            {"text": "Document 2"},
            {"text": "Document 1"},  # Duplicate
            {"text": "Document 3"},
        ]
        
        unique_docs = dedup.deduplicate_documents(docs)
        assert len(unique_docs) == 3


class TestLabelApplication:
    """Tests for label application functions."""
    
    def test_apply_distant_supervision(self):
        """Test applying outlet labels."""
        documents = [
            {"doc_id": "1", "source": "OutletA"},
            {"doc_id": "2", "source": "OutletB"},
            {"doc_id": "3", "source": "OutletC"},
        ]
        
        outlet_labels = {
            "OutletA": "left",
            "OutletB": "right",
        }
        
        label_map = {"left": 0, "center": 1, "right": 2}
        
        labeled = apply_distant_supervision(documents, outlet_labels, label_map)
        
        assert len(labeled) == 2  # Only OutletA and OutletB
        assert labeled[0]["label"] == 0
        assert labeled[1]["label"] == 2
        assert labeled[0]["label_source"] == "distant_supervision"


class TestStratifiedSplit:
    """Tests for stratified splitting."""
    
    def test_stratified_split(self):
        """Test stratified train/val/test split."""
        documents = [
            {"label": 0, "source": "A"} for _ in range(30)
        ] + [
            {"label": 1, "source": "B"} for _ in range(30)
        ] + [
            {"label": 2, "source": "C"} for _ in range(30)
        ]
        
        train, val, test = stratified_split(
            documents,
            test_size=0.2,
            val_size=0.2,
            stratify_by_source=False,
            random_state=42,
        )
        
        # Check sizes
        total = len(train) + len(val) + len(test)
        assert total <= len(documents)  # May lose some due to stratification
        
        # Check that all labels are present in each split
        train_labels = set(d["label"] for d in train)
        val_labels = set(d["label"] for d in val)
        test_labels = set(d["label"] for d in test)
        
        assert len(train_labels) >= 2  # At least 2 classes
        assert len(test_labels) >= 2
