"""
Text preprocessing and data cleaning utilities.

Handles deduplication, language detection, normalization, and tokenization.
"""

import hashlib
import logging
import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Handles text cleaning and normalization."""
    
    def __init__(
        self,
        min_length: int = 50,
        max_length: int = 10000,
        language: str = "en",
        remove_urls: bool = True,
        remove_emails: bool = True,
    ):
        """
        Initialize preprocessor.
        
        Args:
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters
            language: Target language code (only 'en' supported)
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
        """
        self.min_length = min_length
        self.max_length = max_length
        self.language = language
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        
        # Common boilerplate patterns to remove
        self.boilerplate_patterns = [
            r"This article originally appeared on",
            r"Click here to subscribe",
            r"Sign up for our newsletter",
            r"Follow us on Twitter",
            r"Copyright \d{4}",
            r"All rights reserved",
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(
                r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                "",
                text,
            )
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r"\S+@\S+", "", text)
        
        # Remove boilerplate patterns
        for pattern in self.boilerplate_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()
        
        return text
    
    def is_valid_length(self, text: str) -> bool:
        """Check if text meets length requirements."""
        length = len(text)
        return self.min_length <= length <= self.max_length
    
    def detect_language(self, text: str) -> str:
        """
        Simple English language detection.
        
        Args:
            text: Text to check
            
        Returns:
            Language code ('en' or 'other')
        """
        # Simple heuristic: check for common English words
        common_english = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "i", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my", "one",
            "all", "would", "there", "their", "what", "so", "up", "out",
        }
        
        # Tokenize and check proportion of English words
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return "other"
        
        english_count = sum(1 for w in words[:100] if w in common_english)
        proportion = english_count / min(len(words), 100)
        
        return "en" if proportion > 0.2 else "other"
    
    def process_document(self, doc: Dict) -> Optional[Dict]:
        """
        Process a single document.
        
        Args:
            doc: Document dictionary with 'text' and 'title' fields
            
        Returns:
            Processed document or None if invalid
        """
        text = doc.get("text", "")
        title = doc.get("title", "")
        
        # Clean text and title
        clean_text = self.clean_text(text)
        clean_title = self.clean_text(title)
        
        # Combine title and text
        full_text = f"{clean_title}. {clean_text}" if clean_title else clean_text
        
        # Validate length
        if not self.is_valid_length(full_text):
            logger.debug(f"Document {doc.get('doc_id')} rejected: invalid length")
            return None
        
        # Check language
        if self.language == "en" and self.detect_language(full_text) != "en":
            logger.debug(f"Document {doc.get('doc_id')} rejected: not English")
            return None
        
        # Create processed document
        processed = doc.copy()
        processed["text_clean"] = full_text
        processed["text_length"] = len(full_text)
        
        return processed


class Deduplicator:
    """Handles document deduplication using content hashing."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        """
        Initialize deduplicator.
        
        Args:
            similarity_threshold: Threshold for considering documents duplicates
        """
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
    
    def compute_hash(self, text: str) -> str:
        """
        Compute content hash for text.
        
        Args:
            text: Text to hash
            
        Returns:
            Hash string
        """
        # Normalize text for hashing
        normalized = re.sub(r"\s+", " ", text.lower().strip())
        return hashlib.sha256(normalized.encode()).hexdigest()
    
    def is_duplicate(self, text: str) -> bool:
        """
        Check if text is a duplicate.
        
        Args:
            text: Text to check
            
        Returns:
            True if duplicate, False otherwise
        """
        text_hash = self.compute_hash(text)
        
        if text_hash in self.seen_hashes:
            return True
        
        self.seen_hashes.add(text_hash)
        return False
    
    def deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Remove duplicate documents.
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Deduplicated list of documents
        """
        self.seen_hashes.clear()
        unique_docs = []
        
        for doc in documents:
            text = doc.get("text_clean", doc.get("text", ""))
            if not self.is_duplicate(text):
                unique_docs.append(doc)
        
        logger.info(
            f"Deduplication: {len(documents)} -> {len(unique_docs)} "
            f"({len(documents) - len(unique_docs)} duplicates removed)"
        )
        
        return unique_docs


def apply_distant_supervision(
    documents: List[Dict],
    outlet_labels: Dict[str, str],
    label_map: Dict[str, int],
) -> List[Dict]:
    """
    Apply outlet-level labels to documents (distant supervision).
    
    Args:
        documents: List of document dictionaries
        outlet_labels: Mapping from outlet name to bias label
        label_map: Mapping from bias label to numeric class
        
    Returns:
        Documents with added 'label' field
    """
    labeled_docs = []
    
    for doc in documents:
        source = doc.get("source", "")
        if source in outlet_labels:
            bias_label = outlet_labels[source]
            if bias_label in label_map:
                doc["label"] = label_map[bias_label]
                doc["label_source"] = "distant_supervision"
                labeled_docs.append(doc)
    
    logger.info(
        f"Distant supervision: {len(labeled_docs)}/{len(documents)} "
        f"documents labeled"
    )
    
    return labeled_docs


def apply_manual_labels(
    documents: List[Dict],
    manual_labels: Dict[str, int],
    override: bool = True,
) -> List[Dict]:
    """
    Apply manual labels to documents.
    
    Args:
        documents: List of document dictionaries
        manual_labels: Mapping from doc_id to numeric class
        override: Whether to override existing labels
        
    Returns:
        Documents with updated labels
    """
    count = 0
    
    for doc in documents:
        doc_id = doc.get("doc_id", "")
        if doc_id in manual_labels:
            if override or "label" not in doc:
                doc["label"] = manual_labels[doc_id]
                doc["label_source"] = "manual"
                count += 1
    
    logger.info(f"Manual labels: {count} documents labeled/updated")
    
    return documents


def stratified_split(
    documents: List[Dict],
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify_by_source: bool = True,
    random_state: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split documents into train/val/test with stratification.
    
    Args:
        documents: List of labeled document dictionaries
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining after test)
        stratify_by_source: Whether to stratify by source in addition to label
        random_state: Random seed
        
    Returns:
        Tuple of (train_docs, val_docs, test_docs)
    """
    df = pd.DataFrame(documents)
    
    # Create stratification key
    if stratify_by_source and "source" in df.columns:
        df["strat_key"] = df["label"].astype(str) + "_" + df["source"].astype(str)
        # Filter out strat_keys with only 1 sample
        key_counts = df["strat_key"].value_counts()
        valid_keys = key_counts[key_counts >= 2].index
        df = df[df["strat_key"].isin(valid_keys)]
        stratify_col = df["strat_key"]
    else:
        stratify_col = df["label"]
    
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=stratify_col,
        random_state=random_state,
    )
    
    # Second split: train vs val
    if stratify_by_source and "strat_key" in train_val_df.columns:
        stratify_col_2 = train_val_df["strat_key"]
        # Refilter
        key_counts_2 = train_val_df["strat_key"].value_counts()
        valid_keys_2 = key_counts_2[key_counts_2 >= 2].index
        train_val_df = train_val_df[train_val_df["strat_key"].isin(valid_keys_2)]
        stratify_col_2 = train_val_df["strat_key"]
    else:
        stratify_col_2 = train_val_df["label"]
    
    val_size_adjusted = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=stratify_col_2,
        random_state=random_state,
    )
    
    # Remove temporary stratification key
    if "strat_key" in train_df.columns:
        train_df = train_df.drop(columns=["strat_key"])
        val_df = val_df.drop(columns=["strat_key"])
        test_df = test_df.drop(columns=["strat_key"])
    
    logger.info(
        f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    
    return (
        train_df.to_dict("records"),
        val_df.to_dict("records"),
        test_df.to_dict("records"),
    )
