"""
Tests for models module.
"""

import pytest
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from src.models import BaselineClassifier


class TestBaselineClassifier:
    """Tests for BaselineClassifier class."""
    
    def test_initialization(self):
        """Test classifier initialization."""
        # Logistic regression
        clf = BaselineClassifier(classifier_type='logistic', C=1.0)
        assert clf.classifier_type == 'logistic'
        assert clf.model is None  # Not trained yet
        
        # SVM
        clf_svm = BaselineClassifier(classifier_type='svm', C=0.5)
        assert clf_svm.classifier_type == 'svm'
        
        # Invalid type
        with pytest.raises(ValueError):
            BaselineClassifier(classifier_type='invalid')
    
    def test_train(self):
        """Test classifier training."""
        # Generate synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=20,
            n_classes=3,
            random_state=42,
        )
        X_train = csr_matrix(X[:80])
        y_train = y[:80]
        X_val = csr_matrix(X[80:])
        y_val = y[80:]
        
        # Train without calibration
        clf = BaselineClassifier(
            classifier_type='logistic',
            calibration=None,
            random_state=42,
        )
        clf.train(X_train, y_train)
        
        assert clf.model is not None
        assert clf.num_classes == 3
        
        # Train with calibration
        clf_cal = BaselineClassifier(
            classifier_type='logistic',
            calibration='isotonic',
            random_state=42,
        )
        clf_cal.train(X_train, y_train, X_val, y_val)
        
        assert clf_cal.model is not None
    
    def test_predict(self):
        """Test prediction."""
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=20,
            n_classes=3,
            random_state=42,
        )
        X_train = csr_matrix(X[:80])
        y_train = y[:80]
        X_test = csr_matrix(X[80:])
        
        clf = BaselineClassifier(random_state=42)
        clf.train(X_train, y_train)
        
        predictions = clf.predict(X_test)
        
        assert predictions.shape[0] == X_test.shape[0]
        assert predictions.min() >= 0
        assert predictions.max() <= 2
    
    def test_predict_proba(self):
        """Test probability prediction."""
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=20,
            n_classes=3,
            random_state=42,
        )
        X_train = csr_matrix(X[:80])
        y_train = y[:80]
        X_test = csr_matrix(X[80:])
        
        clf = BaselineClassifier(random_state=42)
        clf.train(X_train, y_train)
        
        probas = clf.predict_proba(X_test)
        
        assert probas.shape[0] == X_test.shape[0]
        assert probas.shape[1] == 3
        
        # Check probabilities sum to 1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-5)
        
        # Check probabilities are in valid range
        assert probas.min() >= 0
        assert probas.max() <= 1
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        X, y = make_classification(
            n_samples=100,
            n_features=50,
            n_informative=20,
            n_classes=3,
            random_state=42,
        )
        X_train = csr_matrix(X[:80])
        y_train = y[:80]
        
        feature_names = [f"feature_{i}" for i in range(50)]
        
        clf = BaselineClassifier(random_state=42)
        clf.train(X_train, y_train)
        
        importance = clf.get_feature_importance(feature_names, top_n=10)
        
        # Check structure
        assert len(importance) == 3  # 3 classes
        for class_idx in range(3):
            assert class_idx in importance
            assert len(importance[class_idx]) <= 10
            
            # Check format
            for feature, coef in importance[class_idx]:
                assert isinstance(feature, str)
                assert isinstance(coef, (int, float))


# Note: TransformerClassifier tests would require loading models and are more complex
# For smoke testing in CI, we focus on baseline model tests
# Full transformer tests should be run separately with appropriate fixtures
