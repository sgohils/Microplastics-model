"""Tests for evaluation metrics."""
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.evaluation.metrics import compute_metrics, compare_models
from src.evaluation.uncertainty import DeepEnsemble
from src.evaluation.statistical_tests import (
    paired_t_test,
    permutation_test,
    bootstrap_confidence_interval
)


@pytest.fixture
def predictions():
    """Create sample predictions and true values."""
    np.random.seed(42)
    y_true = np.random.lognormal(0, 1, 100)
    y_pred = y_true + np.random.randn(100) * 0.5
    return y_true, y_pred


class TestComputeMetrics:
    def test_returns_all_metrics(self, predictions):
        y_true, y_pred = predictions
        metrics = compute_metrics(y_true, y_pred)
        
        expected_keys = ['mae', 'rmse', 'r2', 'mape', 'mdae']
        for key in expected_keys:
            assert key in metrics
    
    def test_perfect_predictions(self):
        y = np.random.randn(100)
        metrics = compute_metrics(y, y)
        assert metrics['mae'] == 0
        assert metrics['rmse'] == 0
        assert abs(metrics['r2'] - 1.0) < 1e6  # Perfect R² is 1


class TestBootstrapCI:
    def test_bootstrap_ci_shape(self, predictions):
        y_true, y_pred = predictions
        ci = bootstrap_confidence_interval(y_true, y_pred, n_bootstrap=100, ci=0.95)
        assert 'mae' in ci
        assert 'rmse' in ci
        assert ci['mae']['lower'] <= ci['mae']['mean'] <= ci['mae']['upper']


class TestPermutationTest:
    def test_permutation_test_pvalue(self, predictions):
        y_true, y_pred = predictions
        p_value = permutation_test(y_true, y_true, n_permutations=100)
        assert 0 <= p_value <= 1


class TestPairedTTest:
    def test_paired_t_test(self, predictions):
        y_true, y_pred = predictions
        y_pred2 = y_pred + np.random.randn(100) * 0.1
        
        result = paired_t_test(y_true, y_pred, y_pred2)
        assert 'statistic' in result
        assert 'p_value' in result


class TestCompareModels:
    def test_compare_models(self):
        results = {
            'model_a': {'rmse': 0.5, 'mae': 0.3},
            'model_b': {'rmse': 0.6, 'mae': 0.4},
        }
        comparison = compare_models(results, metric='rmse')
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
