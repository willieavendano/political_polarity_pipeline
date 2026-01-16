"""
Tests for privacy module (k-anonymity enforcement).
"""

import pytest
import pandas as pd
import numpy as np
from src.privacy import KAnonymityEnforcer, AggregateReporter, validate_subset_metadata


class TestKAnonymityEnforcer:
    """Tests for KAnonymityEnforcer class."""
    
    def test_initialization(self):
        """Test enforcer initialization."""
        enforcer = KAnonymityEnforcer(min_group_size=30)
        assert enforcer.min_group_size == 30
        
        # Test invalid k
        with pytest.raises(ValueError):
            KAnonymityEnforcer(min_group_size=1)
    
    def test_check_group_size(self):
        """Test group size checking."""
        enforcer = KAnonymityEnforcer(min_group_size=30)
        
        assert enforcer.check_group_size(50)
        assert enforcer.check_group_size(30)
        assert not enforcer.check_group_size(29)
        assert not enforcer.check_group_size(10)
    
    def test_filter_groups(self):
        """Test filtering of small groups."""
        enforcer = KAnonymityEnforcer(min_group_size=10)
        
        data = pd.DataFrame({
            'region': ['A'] * 15 + ['B'] * 5 + ['C'] * 12,
            'value': range(32),
        })
        
        filtered, suppressed = enforcer.filter_groups(data, 'region')
        
        # Region B should be suppressed (only 5 samples)
        assert len(filtered) == 27  # 15 + 12
        assert 'B' in suppressed
        assert len(suppressed) == 1
        
        # Check that only valid regions remain
        assert set(filtered['region'].unique()) == {'A', 'C'}
    
    def test_aggregate_statistics(self):
        """Test aggregate statistics computation."""
        enforcer = KAnonymityEnforcer(min_group_size=5)
        
        data = pd.DataFrame({
            'region': ['A'] * 10 + ['B'] * 3 + ['C'] * 8,
            'score': np.random.randn(21),
            'value': np.random.randn(21),
        })
        
        aggregated = enforcer.aggregate_statistics(
            data,
            'region',
            ['score', 'value'],
        )
        
        # Region B should be suppressed (only 3 samples)
        assert 'B' not in aggregated.index
        assert 'A' in aggregated.index
        assert 'C' in aggregated.index


class TestAggregateReporter:
    """Tests for AggregateReporter class."""
    
    def test_generate_polarity_report(self):
        """Test polarity report generation."""
        enforcer = KAnonymityEnforcer(min_group_size=5)
        reporter = AggregateReporter(enforcer)
        
        # Create sample predictions
        predictions = pd.DataFrame({
            'region': ['North'] * 10 + ['South'] * 2 + ['East'] * 8,
            'predicted_class': [0, 0, 1, 1, 2, 2, 0, 1, 2, 0] + [0, 1] + [1, 1, 2, 2, 2, 0, 0, 1],
            'polarity_score': np.random.uniform(-1, 1, 20),
            'uncertainty': np.random.uniform(0, 1, 20),
        })
        
        class_names = ['Left', 'Center', 'Right']
        
        report = reporter.generate_polarity_report(
            predictions,
            'region',
            class_names,
        )
        
        # Check structure
        assert 'subset_key' in report
        assert report['subset_key'] == 'region'
        assert 'subsets' in report
        assert 'suppressed_groups' in report
        
        # South should be suppressed (only 2 samples)
        assert 'South' in report['suppressed_groups']
        
        # North and East should be included
        assert 'North' in report['subsets']
        assert 'East' in report['subsets']
        
        # Check subset statistics
        north_stats = report['subsets']['North']
        assert 'count' in north_stats
        assert north_stats['count'] == 10
        assert 'polarity_score' in north_stats
        assert 'class_distribution' in north_stats
        
        # Check class distribution
        for class_name in class_names:
            assert class_name in north_stats['class_distribution']
    
    def test_generate_phrase_report(self):
        """Test phrase report generation."""
        enforcer = KAnonymityEnforcer(min_group_size=5)
        reporter = AggregateReporter(enforcer)
        
        # Create sample predictions with keywords
        predictions = pd.DataFrame({
            'region': ['North'] * 10 + ['South'] * 2 + ['East'] * 8,
            'keywords': [
                ['word1', 'word2'] for _ in range(10)
            ] + [
                ['word3'] for _ in range(2)
            ] + [
                ['word1', 'word3'] for _ in range(8)
            ],
        })
        
        report = reporter.generate_phrase_report(
            predictions,
            'region',
            top_n=5,
        )
        
        # Check structure
        assert 'subset_key' in report
        assert 'subsets' in report
        
        # South should be suppressed
        assert 'South' in report['suppressed_groups']
        
        # Check phrase counts
        assert 'North' in report['subsets']
        north_stats = report['subsets']['North']
        assert 'top_phrases' in north_stats
        assert isinstance(north_stats['top_phrases'], dict)


class TestValidateSubsetMetadata:
    """Tests for subset metadata validation."""
    
    def test_validate_subset_metadata(self):
        """Test validation of subset metadata."""
        documents = [
            {
                'doc_id': '1',
                'subset_meta': {
                    'region': 'North',
                    'age_bucket': '25-34',
                    'forbidden_field': 'value',
                }
            },
            {
                'doc_id': '2',
                'subset_meta': {
                    'region': 'South',
                }
            },
            {
                'doc_id': '3',
                'text': 'No metadata',
            }
        ]
        
        allowed_keys = ['region', 'age_bucket']
        validated = validate_subset_metadata(documents, allowed_keys)
        
        # Check that forbidden_field was removed
        assert 'forbidden_field' not in validated[0]['subset_meta']
        assert 'region' in validated[0]['subset_meta']
        assert 'age_bucket' in validated[0]['subset_meta']
        
        # Check second document
        assert 'region' in validated[1]['subset_meta']
        
        # Check third document (no metadata)
        assert 'subset_meta' in validated[2]
