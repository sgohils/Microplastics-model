"""Evaluation metrics and functions."""
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error, explained_variance_score,
    median_absolute_error
)
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Ensure 1D arrays
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    # Primary metrics
    metrics['mae'] = float(mean_absolute_error(y_true, y_pred))
    metrics['rmse'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics['r2'] = float(r2_score(y_true, y_pred))
    metrics['median_ae'] = float(median_absolute_error(y_true, y_pred))
    
    # Secondary metrics
    metrics['explained_variance'] = float(explained_variance_score(y_true, y_pred))
    
    # MAPE - handle zero values
    if np.any(y_true == 0):
        # Replace zeros with small value for MAPE calculation
        y_true_safe = np.where(y_true == 0, 1e-10, y_true)
        metrics['mape'] = float(mean_absolute_percentage_error(y_true_safe, y_pred))
    else:
        metrics['mape'] = float(mean_absolute_percentage_error(y_true, y_pred))
    
    # Additional diagnostics
    residuals = y_true - y_pred
    metrics['mean_residual'] = float(np.mean(residuals))
    metrics['std_residuals'] = float(np.std(residuals))
    
    return metrics


def compute_interval_metrics(y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            lower: np.ndarray,
                            upper: np.ndarray) -> Dict[str, float]:
    """Compute uncertainty interval metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values (mean)
        lower: Lower prediction bound
        upper: Upper prediction bound
    
    Returns:
        Dictionary of interval metrics
    """
    metrics = {}
    
    # Coverage
    within_interval = (y_true >= lower) & (y_true <= upper)
    metrics['coverage_95'] = float(np.mean(within_interval))
    
    # Interval width
    interval_width = upper - lower
    metrics['mean_interval_width'] = float(np.mean(interval_width))
    metrics['median_interval_width'] = float(np.median(interval_width))
    
    # Continuous Ranked Probability Score (approximation)
    residuals = y_true - y_pred
    std_pred = (upper - lower) / (2 * 1.96)  # Approximate std from interval
    
    # CRPS approximation for Gaussian distribution
    crps = np.abs(residuals) + (2 * std_pred) * (
        1/(2*np.pi) * np.exp(-residuals**2 / (2 * std_pred**2))
    ) - residuals**2 / (2 * std_pred)
    metrics['crps'] = float(np.mean(crps))
    
    # Calibration slope (for uncertainty quality)
    metrics['uncertainty_efficiency'] = float(
        np.exp(np.mean(np.log(interval_width)))
    )
    
    return metrics


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    y_log_true: Optional[np.ndarray] = None,
    y_log_pred: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Comprehensive evaluation of model predictions.
    
    Args:
        y_true: True values (original scale)
        y_pred: Predicted values (original scale)
        lower: Lower prediction interval (original scale, optional)
        upper: Upper prediction interval (original scale, optional)
        y_log_true: True values (log scale, optional)
        y_log_pred: Predicted values (log scale, optional)
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Primary metrics on original scale
    results['original_scale'] = compute_metrics(y_true, y_pred)
    
    # Metrics on log scale if provided
    if y_log_true is not None and y_log_pred is not None:
        results['log_scale'] = compute_metrics(y_log_true, y_log_pred)
    
    # Uncertainty metrics if intervals provided
    if lower is not None and upper is not None:
        results['uncertainty'] = compute_interval_metrics(
            y_true, y_pred, lower, upper
        )
    
    # Error analysis
    residuals = np.array(y_true) - np.array(y_pred)
    results['error_analysis'] = {
        'mean_error': float(np.mean(residuals)),
        'bias': float(np.mean(residuals)),
        'rmse': float(np.sqrt(np.mean(residuals**2))),
        'mae': float(np.mean(np.abs(residuals))),
        'max_abs_error': float(np.max(np.abs(residuals))),
        'outlier_count': int(np.sum(np.abs(residuals) > 2 * np.std(residuals)))
    }
    
    # Relative metrics
    if np.abs(y_true).max() > 0:
        results['relative'] = {
            'nrmse': float(np.sqrt(mean_squared_error(y_true, y_pred)) / 
                          (np.max(y_true) - np.min(y_true))),
            'nmae': float(mean_absolute_error(y_true, y_pred) /
                         (np.max(y_true) - np.min(y_true))),
        }
    
    return results


def compare_models(results: Dict[str, Dict], 
                   metric: str = 'rmse') -> pd.DataFrame:
    """Compare model results.
    
    Args:
        results: Dictionary mapping model names to result dictionaries
        metric: Metric to compare on
    
    Returns:
        Comparison dataframe
    """
    comparison = []
    
    for model_name, model_results in results.items():
        row = {'model': model_name}
        
        if isinstance(model_results, dict):
            if metric in model_results:
                row[metric] = model_results[metric]
            elif 'original_scale' in model_results and metric in model_results['original_scale']:
                row[metric] = model_results['original_scale'][metric]
            
            # Add other metrics
            for key, val in model_results.items():
                if isinstance(val, (int, float)):
                    row[key] = val
        
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df = df.sort_values(metric)
    
    return df


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    metric_fn = mean_squared_error
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        metric_fn: Metric function
    
    Returns:
        Tuple of (metric_value, lower_bound, upper_bound)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    
    # Original metric
    if metric_fn == mean_squared_error:
        orig_metric = np.sqrt(metric_fn(y_true, y_pred))
    else:
        orig_metric = metric_fn(y_true, y_pred)
    
    # Bootstrap
    bootstrap_metrics = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        if metric_fn == mean_squared_error:
            bootstrap_metrics.append(np.sqrt(metric_fn(y_true_boot, y_pred_boot)))
        else:
            bootstrap_metrics.append(metric_fn(y_true_boot, y_pred_boot))
    
    # Compute CI
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
    
    return orig_metric, float(lower), float(upper)


def permutation_test(
    y_true: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
    n_permutations: int = 10000,
    metric: str = 'rmse'
) -> Tuple[float, float]:
    """Perform paired permutation test between two models.
    
    Args:
        y_true: True values
        y_pred_1: Predictions from model 1
        y_pred_2: Predictions from model 2
        n_permutations: Number of permutations
        metric: Metric to compare ('rmse', 'mae')
    
    Returns:
        Tuple of (p_value, observed_difference)
    """
    y_true = np.asarray(y_true)
    y_pred_1 = np.asarray(y_pred_1)
    y_pred_2 = np.asarray(y_pred_2)
    
    # Compute errors for each model
    if metric == 'rmse':
        errors_1 = (y_true - y_pred_1) ** 2
        errors_2 = (y_true - y_pred_2) ** 2
    elif metric == 'mae':
        errors_1 = np.abs(y_true - y_pred_1)
        errors_2 = np.abs(y_true - y_pred_2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Observed difference
    observed_diff = np.mean(errors_1) - np.mean(errors_2)
    
    # Permutation test
    rng = np.random.RandomState(42)
    n = len(errors_1)
    count = 0
    
    for _ in range(n_permutations):
        # Randomly swap errors between models
        swap_mask = rng.random(n) < 0.5
        perm_errors_1 = np.where(swap_mask, errors_1, errors_2)
        perm_errors_2 = np.where(swap_mask, errors_2, errors_1)
        
        perm_diff = np.mean(perm_errors_1) - np.mean(perm_errors_2)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    
    return float(p_value), float(observed_diff)
