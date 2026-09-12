"""Statistical tests for model comparison."""
import numpy as np
from scipy import stats
from sklearn.utils import resample
from typing import Dict, Any, List, Tuple, Optional
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def paired_t_test(y_true: np.ndarray,
                  y_pred1: np.ndarray,
                  y_pred2: np.ndarray,
                  metric: str = 'rmse') -> Dict[str, float]:
    """Perform paired t-test on model errors.
    
    Args:
        y_true: True values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        metric: Metric to compare ('rmse' or 'mae')
    
    Returns:
        Dictionary with test results
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    
    if metric == 'rmse':
        errors1 = (y_true - y_pred1) ** 2
        errors2 = (y_true - y_pred2) ** 2
        metric_name = 'MSE'
        metric_value_fn = lambda e: np.sqrt(np.mean(e))
    elif metric == 'mae':
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
        metric_name = 'MAE'
        metric_value_fn = lambda e: np.mean(e)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Paired differences
    diffs = errors1 - errors2
    
    # T-test
    t_stat, p_value = stats.ttest_rel(errors1, errors2)
    
    # Effect size (Cohen's d)
    cohens_d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs, ddof=1) > 0 else 0
    
    results = {
        'metric': metric_name,
        'model1_error': metric_value_fn(errors1),
        'model2_error': metric_value_fn(errors2),
        'mean_diff': float(np.mean(diffs)),
        'std_diff': float(np.std(diffs, ddof=1)),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significant': bool(p_value < 0.05),
        'n_samples': len(y_true)
    }
    
    return results


def permutation_test(y_true: np.ndarray,
                     y_pred1: np.ndarray,
                     y_pred2: np.ndarray,
                     n_permutations: int = 10000,
                     metric: str = 'rmse',
                     seed: int = 42) -> Dict[str, float]:
    """Perform permutation test on model differences.
    
    Args:
        y_true: True values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        n_permutations: Number of permutations
        metric: Metric to compare
        seed: Random seed
    
    Returns:
        Dictionary with test results
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    
    rng = np.random.RandomState(seed)
    
    if metric == 'rmse':
        errors1 = (y_true - y_pred1) ** 2
        errors2 = (y_true - y_pred2) ** 2
    elif metric == 'mae':
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Observed difference
    observed_diff = np.mean(errors1) - np.mean(errors2)
    
    # Permutation test
    n = len(errors1)
    count = 0
    
    for _ in range(n_permutations):
        swap_mask = rng.random(n) < 0.5
        perm_errors1 = np.where(swap_mask, errors1, errors2)
        perm_errors2 = np.where(swap_mask, errors2, errors1)
        
        perm_diff = np.mean(perm_errors1) - np.mean(perm_errors2)
        
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    
    results = {
        'metric': metric.upper(),
        'model1_error': float(np.sqrt(np.mean(errors1)) if metric == 'rmse' else np.mean(errors1)),
        'model2_error': float(np.sqrt(np.mean(errors2)) if metric == 'rmse' else np.mean(errors2)),
        'observed_difference': float(observed_diff),
        'p_value': float(p_value),
        'n_permutations': n_permutations,
        'significant': bool(p_value < 0.05),
        'n_samples': n
    }
    
    return results


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    metric: str = 'rmse'
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a model's error.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        metric: Metric to bootstrap
    
    Returns:
        Dictionary with CI and point estimate
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    rng = np.random.RandomState(42)
    
    metric_values = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if metric == 'rmse':
            metric_values.append(np.sqrt(np.mean((y_true_boot - y_pred_boot) ** 2)))
        elif metric == 'mae':
            metric_values.append(np.mean(np.abs(y_true_boot - y_pred_boot)))
        elif metric == 'r2':
            from sklearn.metrics import r2_score
            metric_values.append(r2_score(y_true_boot, y_pred_boot))
        elif metric == 'mdae':
            metric_values.append(np.median(np.abs(y_true_boot - y_pred_boot)))
    
    point_estimate = metric_values[len(metric_values) // 2]  # Use middle as approximation
    alpha = 1 - confidence
    lower = np.percentile(metric_values, alpha / 2 * 100)
    upper = np.percentile(metric_values, (1 - alpha / 2) * 100)
    
    return {
        'metric': metric.upper(),
        'point_estimate': float(point_estimate),
        'lower_ci': float(lower),
        'upper_ci': float(upper),
        'confidence_level': confidence,
        'n_bootstrap': n_bootstrap,
        'mean': float(np.mean(metric_values)),
        'std': float(np.std(metric_values))
    }


def bonferroni_correction(p_values: List[float], 
                          alpha: float = 0.05) -> List[float]:
    """Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate
    
    Returns:
        List of adjusted p-values
    """
    n = len(p_values)
    adjusted = [min(1.0, p * n) for p in p_values]
    return adjusted


def compute_effect_size_diffs(y_true: np.ndarray,
                              y_pred1: np.ndarray,
                              y_pred2: np.ndarray,
                              metric: str = 'rmse') -> Dict[str, float]:
    """Compute effect sizes for model comparison.
    
    Args:
        y_true: True values
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        metric: Metric to compare
    
    Returns:
        Dictionary with effect sizes
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)
    
    if metric == 'rmse':
        errors1 = np.sqrt((y_true - y_pred1) ** 2)
        errors2 = np.sqrt((y_true - y_pred2) ** 2)
    elif metric == 'mae':
        errors1 = np.abs(y_true - y_pred1)
        errors2 = np.abs(y_true - y_pred2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Cohen's d
    diff = errors1 - errors2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
    
    # Relative improvement
    model1_error = np.mean(errors1)
    model2_error = np.mean(errors2)
    relative_diff = (model1_error - model2_error) / model2_error if model2_error > 0 else 0
    
    return {
        'cohens_d': float(cohens_d),
        'relative_improvement': float(relative_diff),
        'model1_error': float(model1_error),
        'model2_error': float(model2_error),
        'mean_difference': float(np.mean(diff)),
        'std_difference': float(np.std(diff, ddof=1))
    }


def run_statistical_comparison(
    model_results: Dict[str, Dict[str, np.ndarray]],
    y_true: np.ndarray,
    baseline_model_name: str,
    test_model_name: str
) -> Dict[str, Any]:
    """Run complete statistical comparison between two models.
    
    Args:
        model_results: Dictionary mapping model names to predictions
        y_true: True values
        baseline_model_name: Name of baseline model
        test_model_name: Name of test model
    
    Returns:
        Comprehensive statistical comparison results
    """
    results = {}
    
    y_pred_baseline = model_results[baseline_model_name]['predictions']
    y_pred_test = model_results[test_model_name]['predictions']
    
    # Paired t-tests
    for metric in ['rmse', 'mae']:
        results[f'paired_ttest_{metric}'] = paired_t_test(
            y_true, y_pred_baseline, y_pred_test, metric
        )
    
    # Permutation tests
    for metric in ['rmse', 'mae']:
        results[f'permutation_test_{metric}'] = permutation_test(
            y_true, y_pred_baseline, y_pred_test, 
            metric=metric, seed=42
        )
    
    # Bootstrap CIs
    results['baseline_ci_rmse'] = bootstrap_confidence_interval(
        y_true, y_pred_baseline, metric='rmse'
    )
    results['test_ci_rmse'] = bootstrap_confidence_interval(
        y_true, y_pred_test, metric='rmse'
    )
    
    # Effect sizes
    results['effect_sizes'] = compute_effect_size_diffs(
        y_true, y_pred_baseline, y_pred_test
    )
    
    # Summary
    results['summary'] = {
        'baseline_better': results['permutation_test_rmse']['p_value'] > 0.05 and
                          float(np.mean((y_true - y_pred_baseline)**2)) < 
                          float(np.mean((y_true - y_pred_test)**2)),
        'test_better': results['permutation_test_rmse']['significant'] and
                      float(np.mean((y_true - y_pred_test)**2)) < 
                      float(np.mean((y_true - y_pred_baseline)**2)),
        'no_significant_difference': not results['permutation_test_rmse']['significant']
    }
    
    return results
