"""Sparse data and extreme event experiments."""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from typing import Dict, Any

from src.models.baselines import XGBoostModel, MLPModel
from src.evaluation.metrics import compute_metrics
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_experiment_e17_sparse_data(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E17: Sparse data degradation experiment.
    
    Tests how model performance degrades as training data is reduced.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    
    logger.info(f"Running E17: Sparse data experiment with {len(features)} samples")
    
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
    from sklearn.model_selection import train_test_split
    
    X = features[feature_cols].fillna(0).values
    y = features['target'].values
    
    X_full_train, X_test, y_full_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    sparsity_levels = [1.0, 0.75, 0.50, 0.25, 0.10]
    results = {}
    
    for sparsity in sparsity_levels:
        # Subsample training data
        n_samples = int(len(X_full_train) * sparsity)
        np.random.seed(seed)
        indices = np.random.choice(len(X_full_train), size=n_samples, replace=False)
        
        X_train = X_full_train[indices]
        y_train = y_full_train[indices]
        
        if len(y_train) < 3:
            logger.warning(f"Not enough samples at {sparsity*100}% sparsity")
            continue
        
        # Normalize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'xgboost': XGBoostModel(config),
            'mlp': MLPModel(config),
        }
        
        sparsity_results = {}
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                y_pred_orig = np.expm1(y_pred)
                y_test_orig = np.expm1(y_test)
                
                metrics = compute_metrics(y_test_orig, np.maximum(y_pred_orig, 0))
                sparsity_results[name] = metrics
                
                logger.info(f"  {sparsity*100}%: {name} RMSE={metrics['rmse']:.4f}")
            except Exception as e:
                logger.error(f"  {sparsity*100}%: {name} failed: {e}")
                sparsity_results[name] = {'error': str(e)}
        
        results[f'sparsity_{sparsity}'] = {
            'train_samples': n_samples,
            'results': sparsity_results
        }
    
    # Save results
    results_dir = Path(config['experiments']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'e17_sparse_data.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return {
        'experiment': 'E17_sparse_data',
        'seed': seed,
        'results': results
    }


def run_experiment_e20_extreme_events(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E20: Extreme event analysis.
    
    Compare performance during normal vs. high-flow/high-precipitation events.
    
    Args:
        config: Configuration dictionary
        seed: Random seed
    
    Returns:
        Results dictionary
    """
    set_seed(seed)
    
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    
    logger.info(f"Running E20: Extreme event analysis with {len(features)} samples")
    
    # Identify extreme events based on available features
    results = {}
    
    # High discharge events (top 10%)
    if 'discharge_m3s' in features.columns:
        discharge_threshold = features['discharge_m3s'].quantile(0.9)
        high_flow_mask = features['discharge_m3s'] > discharge_threshold
        normal_flow_mask = ~high_flow_mask
        
        logger.info(f"  High flow events: {high_flow_mask.sum()}")
        logger.info(f"  Normal flow events: {normal_flow_mask.sum()}")
    
    # High precipitation events (> 25mm)
    if 'precipitation_mm' in features.columns:
        extreme_precip_mask = features['precipitation_mm'] > 25
        normal_precip_mask = ~extreme_precip_mask
        
        logger.info(f"  Extreme precipitation events: {extreme_precip_mask.sum()}")
        logger.info(f"  Normal precipitation events: {normal_precip_mask.sum()}")
    
    # Train model
    feature_cols = [c for c in features.columns 
                   if c not in ['target', 'obs_index', 'location_name', 'date',
                               'date_parsed', 'source_dataset', 'watershed',
                               'latitude', 'longitude', 'original_units',
                               'citation', 'notes', 'particle_types', 'basin',
                               'source_url', 'sampling_method', 'nearest_segment_id',
                               'has_max', 'below_detection_limit', 'season',
                               'particle_count']]
    
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    X = features[feature_cols].fillna(0).values
    y = features['target'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train XGBoost
    xgb = XGBoostModel(config)
    xgb.fit(X_train_scaled, y_train)
    y_pred = xgb.predict(X_test_scaled)
    
    # Evaluate by condition
    # For this, use the full features for evaluation
    y_pred_orig = np.expm1(np.maximum(y_pred, 0))
    y_test_orig = np.expm1(y_test)
    
    if 'discharge_m3s' in features.columns:
        test_discharge = features.loc[y_test.index if hasattr(y_test, 'index') else range(len(X_test)), 'discharge_m3s']
        test_discharge_values = features['discharge_m3s'].iloc[
            X_test.shape[0]:X_test.shape[0] + len(y_test)
        ].values if len(features) > X_test.shape[0] else np.full(len(y_test), np.nan)
        
        high_flow_idx = test_discharge_values > discharge_threshold if not np.isnan(test_discharge_values).all() else np.array([])
        
        if len(high_flow_idx) > 0:
            high_metrics = compute_metrics(y_test_orig[high_flow_idx.astype(bool)], 
                                          y_pred_orig[high_flow_idx.astype(bool)])
            results['high_flow'] = high_metrics
        
        results['normal_flow'] = compute_metrics(y_test_orig, y_pred_orig)
    
    results['all_events'] = compute_metrics(y_test_orig, y_pred_orig)
    results['discharge_threshold'] = float(discharge_threshold) if 'discharge_m3s' in features.columns else None
    
    # Save results
    results_dir = Path(config['experiments']['results_dir'])
    with open(results_dir / 'e20_extreme_events.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return {
        'experiment': 'E20_extreme_events',
        'seed': seed,
        'results': results
    }
