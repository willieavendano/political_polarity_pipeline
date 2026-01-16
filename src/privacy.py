"""
Privacy-preserving utilities for subset analysis.

Implements k-anonymity and aggregate-only reporting.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class KAnonymityEnforcer:
    """Enforces k-anonymity for subset analysis."""
    
    def __init__(self, min_group_size: int = 30):
        """
        Initialize k-anonymity enforcer.
        
        Args:
            min_group_size: Minimum group size (k value)
        """
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")
        
        self.min_group_size = min_group_size
        logger.info(f"K-anonymity enforcer initialized with k={min_group_size}")
    
    def check_group_size(self, group_size: int) -> bool:
        """
        Check if a group meets the minimum size requirement.
        
        Args:
            group_size: Size of the group
            
        Returns:
            True if group size is sufficient, False otherwise
        """
        return group_size >= self.min_group_size
    
    def filter_groups(
        self,
        data: pd.DataFrame,
        group_by_column: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Filter out groups that don't meet minimum size.
        
        Args:
            data: DataFrame to filter
            group_by_column: Column to group by
            
        Returns:
            Tuple of (filtered_data, suppressed_groups)
        """
        # Count group sizes
        group_counts = data[group_by_column].value_counts()
        
        # Identify groups to keep and suppress
        valid_groups = group_counts[group_counts >= self.min_group_size].index.tolist()
        suppressed_groups = group_counts[group_counts < self.min_group_size].index.tolist()
        
        # Filter data
        filtered_data = data[data[group_by_column].isin(valid_groups)].copy()
        
        if suppressed_groups:
            logger.warning(
                f"Suppressed {len(suppressed_groups)} groups "
                f"(< {self.min_group_size} samples): {suppressed_groups}"
            )
        
        logger.info(
            f"Retained {len(valid_groups)} groups "
            f"with {len(filtered_data)} total samples"
        )
        
        return filtered_data, suppressed_groups
    
    def aggregate_statistics(
        self,
        data: pd.DataFrame,
        group_by_column: str,
        value_columns: List[str],
    ) -> pd.DataFrame:
        """
        Compute aggregate statistics with k-anonymity.
        
        Args:
            data: DataFrame with data
            group_by_column: Column to group by
            value_columns: Columns to aggregate
            
        Returns:
            DataFrame with aggregate statistics
        """
        # Filter groups first
        filtered_data, _ = self.filter_groups(data, group_by_column)
        
        if filtered_data.empty:
            logger.warning("No groups meet minimum size requirement")
            return pd.DataFrame()
        
        # Compute aggregates
        agg_funcs = {
            col: ['mean', 'median', 'std', 'count']
            for col in value_columns
        }
        
        aggregated = filtered_data.groupby(group_by_column).agg(agg_funcs)
        
        return aggregated


class AggregateReporter:
    """Generates privacy-preserving aggregate reports."""
    
    def __init__(self, k_anonymity: KAnonymityEnforcer):
        """
        Initialize reporter.
        
        Args:
            k_anonymity: K-anonymity enforcer
        """
        self.k_anonymity = k_anonymity
    
    def generate_polarity_report(
        self,
        predictions: pd.DataFrame,
        subset_key: str,
        class_names: List[str],
    ) -> Dict[str, Any]:
        """
        Generate polarity report by subset.
        
        Args:
            predictions: DataFrame with predictions and metadata
            subset_key: Column name for subset grouping
            class_names: List of class names
            
        Returns:
            Dictionary with report data
        """
        logger.info(f"Generating polarity report by '{subset_key}'...")
        
        # Check if subset_key exists
        if subset_key not in predictions.columns:
            logger.error(f"Subset key '{subset_key}' not found in predictions")
            return {}
        
        # Filter groups by k-anonymity
        filtered_data, suppressed = self.k_anonymity.filter_groups(
            predictions, subset_key
        )
        
        if filtered_data.empty:
            logger.warning("No subsets meet k-anonymity requirements")
            return {
                "subset_key": subset_key,
                "subsets": {},
                "suppressed_groups": suppressed,
            }
        
        # Compute statistics by subset
        subsets = {}
        
        for subset_value in filtered_data[subset_key].unique():
            subset_data = filtered_data[filtered_data[subset_key] == subset_value]
            
            # Basic statistics
            stats = {
                "count": len(subset_data),
                "polarity_score": {
                    "mean": float(subset_data["polarity_score"].mean()),
                    "median": float(subset_data["polarity_score"].median()),
                    "std": float(subset_data["polarity_score"].std()),
                },
                "uncertainty": {
                    "mean": float(subset_data["uncertainty"].mean()),
                    "median": float(subset_data["uncertainty"].median()),
                },
                "class_distribution": {},
            }
            
            # Class distribution
            for i, class_name in enumerate(class_names):
                class_count = (subset_data["predicted_class"] == i).sum()
                stats["class_distribution"][class_name] = {
                    "count": int(class_count),
                    "proportion": float(class_count / len(subset_data)),
                }
            
            subsets[str(subset_value)] = stats
        
        report = {
            "subset_key": subset_key,
            "min_group_size": self.k_anonymity.min_group_size,
            "total_subsets": len(subsets),
            "suppressed_groups": suppressed,
            "subsets": subsets,
        }
        
        logger.info(
            f"Report generated: {len(subsets)} subsets, "
            f"{len(suppressed)} suppressed"
        )
        
        return report
    
    def generate_phrase_report(
        self,
        predictions: pd.DataFrame,
        subset_key: str,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate top phrases report by subset.
        
        Args:
            predictions: DataFrame with predictions and keywords
            subset_key: Column name for subset grouping
            top_n: Number of top phrases per subset
            
        Returns:
            Dictionary with phrase report
        """
        logger.info(f"Generating phrase report by '{subset_key}'...")
        
        if subset_key not in predictions.columns:
            logger.error(f"Subset key '{subset_key}' not found in predictions")
            return {}
        
        if "keywords" not in predictions.columns:
            logger.warning("No keywords column found in predictions")
            return {}
        
        # Filter by k-anonymity
        filtered_data, suppressed = self.k_anonymity.filter_groups(
            predictions, subset_key
        )
        
        if filtered_data.empty:
            return {
                "subset_key": subset_key,
                "subsets": {},
                "suppressed_groups": suppressed,
            }
        
        # Extract phrases by subset
        subsets = {}
        
        for subset_value in filtered_data[subset_key].unique():
            subset_data = filtered_data[filtered_data[subset_key] == subset_value]
            
            # Aggregate keywords/phrases
            all_keywords = []
            for keywords in subset_data["keywords"]:
                if isinstance(keywords, list):
                    all_keywords.extend(keywords)
            
            # Count frequency
            keyword_counts = pd.Series(all_keywords).value_counts()
            top_keywords = keyword_counts.head(top_n).to_dict()
            
            subsets[str(subset_value)] = {
                "count": len(subset_data),
                "top_phrases": top_keywords,
            }
        
        report = {
            "subset_key": subset_key,
            "subsets": subsets,
            "suppressed_groups": suppressed,
        }
        
        return report


def validate_subset_metadata(
    documents: List[Dict],
    allowed_keys: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Validate and clean subset metadata.
    
    Ensures only externally provided metadata is used, never inferred.
    
    Args:
        documents: List of document dictionaries
        allowed_keys: Optional list of allowed metadata keys
        
    Returns:
        Validated documents
    """
    if allowed_keys is None:
        # Default allowed keys (non-sensitive, coarse-grained)
        allowed_keys = [
            "region",
            "state",
            "country",
            "year",
            "month",
            "age_bucket",
            "source_type",
        ]
    
    validated = []
    
    for doc in documents:
        if "subset_meta" in doc and isinstance(doc["subset_meta"], dict):
            # Filter to allowed keys only
            filtered_meta = {
                k: v for k, v in doc["subset_meta"].items()
                if k in allowed_keys
            }
            doc["subset_meta"] = filtered_meta
        
        validated.append(doc)
    
    return validated


def create_privacy_notice() -> str:
    """
    Create a privacy notice for reports.
    
    Returns:
        Privacy notice text
    """
    notice = """
    PRIVACY NOTICE
    ==============
    
    This report contains AGGREGATE-LEVEL analysis only.
    
    - Minimum group size enforced (k-anonymity)
    - No individual-level predictions or targeting
    - Metadata provided externally, never inferred
    - No protected attribute inference
    
    PROHIBITED USES:
    - Individual profiling or targeting
    - Inferring protected characteristics
    - Decisions affecting individual rights
    
    This is an analytical tool for understanding discourse patterns,
    not for classifying individuals.
    """
    return notice
