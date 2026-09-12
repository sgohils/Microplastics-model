"""Spatial and temporal generalization experiments."""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, Any

from src.models.baselines import XGBoostModel
from src.evaluation.metrics import compute_metrics
from src.utils.config import load_config
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_experiment_e11_spatial_holdout(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E11: Spatial holdout generalization.
    
    Train on some sub-watersheds, test on held-out sub-watersheds.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    
    logger.info(f"Running E11: Spatial holdout with {len(features)} samples")
    
    # Split by watershed (spatial holdout)
    watersheds = features['watershed'].unique()
    n_holdout = max(1, len(watersheds) // 4)
    
    np.random.seed(seed)
    holdout_watersheds = np.random.choice(watersheds, size=n_holdout, replace=False)
    
    train_mask = ~features['watershed'].isin(holdout_watersheds)
    test_mask = features['watershed'].isin(holdout_watersheds)
    
    logger.info(f"Spatial holdout watersheds: {holdout_watersheds}")
    logger.info(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")
    
    # Prepare data
    feature_cols = [c for c in features.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    from sklearn.preprocessing import StandardScaler
    
    X = features[feature_cols].fillna(0).values
    y = features['target'].values
    
    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    
    # Normalize (fit only on training)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb = XGBoostModel(config)
    xgb.fit(X_train_scaled, y_train)
    y_pred = xgb.predict(X_test_scaled)
    
    # Transform back
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, np.maximum(y_pred_orig, 0))
    
    results = {
        'experiment': 'E11_spatial_holdout',
        'seed': seed,
        'holdout_watersheds': list(holdout_watersheds),
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'metrics': metrics,
        'model': 'XGBoost'
    }
    
    logger.info(f"  Spatial holdout RMSE: {metrics['rmse']:.4f}, "
               f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return results


def run_experiment_e12_temporal_holdout(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E12: Temporal holdout generalization.
    
    Train on earlier dates, test on later dates.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    
    logger.info(f"Running E12: Temporal holdout with {len(features)} samples")
    
    # Split by date
    features['date_parsed'] = pd.to_datetime(features['date'])
    
    # Sort by date
    features_sorted = features.sort_values('date_parsed')
    
    # Find median date
    mid_date = features_sorted['date_parsed'].median()
    
    train_mask = features_sorted['date_parsed'] < mid_date
    test_mask = features_sorted['date_parsed'] >= mid_date
    
    logger.info(f"Temporal split point: {mid_date}")
    logger.info(f"Train: {train_mask.sum()}, Test: {test_mask.sum()}")
    
    # Prepare features
    feature_cols = [c for c in features_sorted.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    from sklearn.preprocessing import StandardScaler
    
    X = features_sorted[feature_cols].fillna(0).values
    y = features_sorted['target'].values
    
    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb = XGBoostModel(config)
    xgb.fit(X_train_scaled, y_train)
    y_pred = xgb.predict(X_test_scaled)
    
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, np.maximum(y_pred_orig, 0))
    
    results = {
        'experiment': 'E12_temporal_holdout',
        'seed': seed,
        'split_date': str(mid_date),
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'metrics': metrics,
        'model': 'XGBoost'
    }
    
    logger.info(f"  Temporal holdout RMSE: {metrics['rmse']:.4f}, "
               f"MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return results


def run_experiment_e13_spatiotemporal_holdout(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E13: Spatiotemporal holdout (hardest generalization).
    
    Hold out both geography AND time.
    """
    set_seed(seed)
    
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    features['date_parsed'] = pd.to_datetime(features['date'])
    
    logger.info(f"Running E13: Spatiotemporal holdout with {len(features)} samples")
    
    # Hold out specific watersheds AND specific time period
    features_sorted = features.sort_values('date_parsed')
    mid_date = features_sorted['date_parsed'].median()
    
    # Find watersheds in the later period
    late_watersheds = features_sorted[
        features_sorted['date_parsed'] >= mid_date
    ]['watershed'].unique()
    
    # Hold out the latest half of watersheds in late period
    np.random.seed(seed)
    holdout_watersheds = np.random.choice(
        late_watersheds, 
        size=max(1, len(late_watersheds) // 2), 
        replace=False
    )
    
    # Training: early period OR non-held-out watersheds in late period
    train_mask = (
        (features_sorted['date_parsed'] < mid_date) |
        (~features_sorted['watershed'].isin(holdout_watersheds))
    )
    
    # Test: held-out watersheds in late period
    test_mask = (
        (features_sorted['date_parsed'] >= mid_date) &
        (features_sorted['watershed'].isin(holdout_watersheds))
    )
    
    logger.info(f"  Holdout watersheds: {holdout_watersheds}")
    logger.info(f"  Train: {train_mask.sum()}, Test: {test_mask.sum()}")
    
    if test_mask.sum() == 0:
        logger.warning("No test samples in spatiotemporal holdout")
        return {'experiment': 'E13', 'error': 'No test samples available'}
    
    # Prepare features
    feature_cols = [c for c in features_sorted.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    from sklearn.preprocessing import StandardScaler
    
    X = features_sorted[feature_cols].fillna(0).values
    y = features_sorted['target'].values
    
    X_train = X[train_mask.values]
    y_train = y[train_mask.values]
    X_test = X[test_mask.values]
    y_test = y[test_mask.values]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb = XGBoostModel(config)
    xgb.fit(X_train_scaled, y_train)
    y_pred = xgb.predict(X_test_scaled)
    
    y_pred_orig = np.expm1(y_pred)
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, np.maximum(y_pred_orig, 0))
    
    results = {
        'experiment': 'E13_spatiotemporal_holdout',
        'seed': seed,
        'split_date': str(mid_date),
        'holdout_watersheds': list(holdout_watersheds),
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'metrics': metrics,
        'model': 'XGBoost'
    }
    
    logger.info(f"  Spatiotemporal holdout RMSE: {metrics['rmse']:.4f}")
    
    return results


def run_experiment_e18_unseen_watershed(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E18: Unseen watershed comparison.
    
    Compare XGBoost, MLP, and GNN on completely unseen watersheds.
    """
    set_seed(seed)
    
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    features['date_parsed'] = pd.to_datetime(features['date'])
    
    logger.info(f"Running E18: Unseen watershed generalization")
    
    # Get unique watersheds
    watersheds = features['watershed'].dropna().unique()
    
    if len(watersheds) < 2:
        logger.warning("Not enough watersheds for unseen-watershed experiment")
        return {'experiment': 'E18', 'error': 'Not enough watersheds'}
    
    # For each watershed, train on all others and test on it
    results = {}
    
    feature_cols = [c for c in features.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    from sklearn.preprocessing import StandardScaler
    
    for test_watershed in watersheds:
        train_data = features[features['watershed'] != test_watershed]
        test_data = features[features['watershed'] == test_watershed]
        
        if len(test_data) < 2 or len(train_data) < 5:
            continue
        
        X_train = train_data[feature_cols].fillna(0).values
        y_train = train_data['target'].values
        X_test = test_data[feature_cols].fillna(0).values
        y_test = test_data['target'].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost
        xgb = XGBoostModel(config)
        xgb.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb.predict(X_test_scaled)
        
        # Train MLP
        mlp = MLPModel(config)
        mlp.fit(X_train_scaled, y_train)
        y_pred_mlp = mlp.predict(X_test_scaled)
        
        # Evaluate
        y_pred_xgb_orig = np.expm1(y_pred_xgb)
        y_pred_mlp_orig = np.expm1(y_pred_mlp)
        y_test_orig = np.expm1(y_test)
        
        xgb_metrics = compute_metrics(y_test_orig, np.maximum(y_pred_xgb_orig, 0))
        mlp_metrics = compute_metrics(y_test_orig, np.maximum(y_pred_mlp_orig, 0))
        
        results[test_watershed] = {
            'xgboost': xgb_metrics,
            'mlp': mlp_metrics,
            'n_train': len(train_data),
            'n_test': len(test_data)
        }
        
        logger.info(f"  {test_watershed}: XGB RMSE={xgb_metrics['rmse']:.4f}, "
                   f"MLP RMSE={mlp_metrics['rmse']:.4f}")
    
    return {
        'experiment': 'E18_unseen_watershed',
        'seed': seed,
        'results': results
    }


def run_experiment_e19_temporal_extrapolation(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E19: Temporal extrapolation test."""
    return run_experiment_e12_temporal_holdout(config, seed)
