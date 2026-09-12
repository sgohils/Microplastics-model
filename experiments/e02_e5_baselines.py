"""E2-E5: Additional baseline experiments."""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any

from src.models.baselines import SVMModel, GaussianProcessModel, MLPModel, EnsembleModel
from src.evaluation.metrics import compute_metrics
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def prepare_data(config: Dict[str, Any]):
    """Load and prepare data for experiments.
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    processed_dir = Path(config['data']['processed_dir'])
    features = pd.read_csv(processed_dir / "features_processed.csv")
    
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
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    X = features[feature_cols].fillna(0).values
    y = features['target'].values
    
    # Apply log1p transformation to target (consistent with E1)
    y = np.log1p(y)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def run_experiment_e2_svm_rbf(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E2: SVM with RBF kernel baseline."""
    set_seed(seed)
    logger.info("Running E2: SVM-RBF baseline")
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(config)
    
    model = SVMModel(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_orig = np.expm1(np.maximum(y_pred, 0))
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, y_pred_orig)
    
    logger.info(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return {
        'experiment': 'E2_svm_rbf',
        'seed': seed,
        'metrics': metrics,
        'model': 'SVM-RBF'
    }


def run_experiment_e3_gaussian_process(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E3: Gaussian Process regression baseline."""
    set_seed(seed)
    logger.info("Running E3: Gaussian Process baseline")
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(config)
    
    model = GaussianProcessModel(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_orig = np.expm1(np.maximum(y_pred, 0))
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, y_pred_orig)
    
    logger.info(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return {
        'experiment': 'E3_gaussian_process',
        'seed': seed,
        'metrics': metrics,
        'model': 'GaussianProcess'
    }


def run_experiment_e4_mlp(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E4: MLP baseline."""
    set_seed(seed)
    logger.info("Running E4: MLP baseline")
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(config)
    
    model = MLPModel(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_orig = np.expm1(np.maximum(y_pred, 0))
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, y_pred_orig)
    
    logger.info(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return {
        'experiment': 'E4_mlp',
        'seed': seed,
        'metrics': metrics,
        'model': 'MLP'
    }


def run_experiment_e5_ensemble(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E5: Ensemble model."""
    set_seed(seed)
    logger.info("Running E5: Ensemble baseline")
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(config)
    
    model = EnsembleModel(config)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    y_pred_orig = np.expm1(np.maximum(y_pred, 0))
    y_test_orig = np.expm1(y_test)
    
    metrics = compute_metrics(y_test_orig, y_pred_orig)
    
    logger.info(f"  RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
    
    return {
        'experiment': 'E5_ensemble',
        'seed': seed,
        'metrics': metrics,
        'model': 'Ensemble'
    }


def run_experiment_e7_feature_importance(
    config: Dict[str, Any],
    seed: int = 42
) -> Dict[str, Any]:
    """Experiment E7: Feature importance analysis."""
    set_seed(seed)
    logger.info("Running E7: Feature importance")
    
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(config)
    
    model = EnsembleModel(config)
    model.fit(X_train, y_train)
    
    feature_names = [c for c in pd.read_csv(
        Path(config['data']['processed_dir']) / "features_processed.csv"
    ).columns if c not in [
        'target', 'obs_index', 'location_name', 'date', 'date_parsed',
        'source_dataset', 'watershed', 'latitude', 'longitude', 'original_units',
        'citation', 'notes', 'particle_types', 'basin', 'source_url',
        'sampling_method', 'nearest_segment_id', 'has_max', 'below_detection_limit',
        'season', 'particle_count'
    ]]
    
    importance = model.feature_importance()
    
    top_features = sorted(
        [(feature_names[i], float(importance[i])) for i in range(len(feature_names))],
        key=lambda x: abs(x[1]), reverse=True
    )[:20]
    
    logger.info(f"  Top 5 features: {top_features[:5]}")
    
    return {
        'experiment': 'E7_feature_importance',
        'seed': seed,
        'top_features': top_features,
        'model': 'Ensemble'
    }
