"""Baseline machine learning models for microplastic prediction."""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model = None
        self.scaler_X: Optional[StandardScaler] = None
        self.scaler_y: Optional[StandardScaler] = None
        self.is_fitted = False
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        """Fit the model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        raise NotImplementedError
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return {}


class MeanPredictor(BaseModel):
    """Mean/median baseline predictor."""
    
    def __init__(self, strategy: str = "median", config: Dict = None):
        super().__init__(config)
        self.strategy = strategy
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model = DummyRegressor(strategy=self.strategy)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return {'train_mae': 0.0, 'val_mae': 0.0}
    
    def predict(self, X):
        return self.model.predict(X)


class RidgeRegressor(BaseModel):
    """Ridge regression baseline."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        alpha = config.get('model', {}).get('ridge_alpha', 1.0) if config else 1.0
        self.model = Ridge(alpha=alpha)
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Fit scaler on training data
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Compute training metrics
        train_pred = self.model.predict(X_train_scaled)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        result = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': r2_score(y_train, train_pred)
        }
        
        # Validation metrics
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            result['val_mae'] = mean_absolute_error(y_val, val_pred)
            result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            result['val_r2'] = r2_score(y_val, val_pred)
        
        return result
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        return self.model.predict(X_scaled)


class RandomForestModel(BaseModel):
    """Random Forest baseline."""
    
    def __init__(self, config: Dict = None, n_estimators: int = 500, 
                 max_depth: int = None):
        super().__init__(config)
        
        params = config.get('model', {}) if config else {}
        self.model = RandomForestRegressor(
            n_estimators=params.get('rf_n_estimators', n_estimators),
            max_depth=params.get('rf_max_depth', max_depth),
            max_features=params.get('rf_max_features', 'sqrt'),
            random_state=42,
            n_jobs=-1
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        result = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            result['val_mae'] = mean_absolute_error(y_val, val_pred)
            result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            result['val_r2'] = r2_score(y_val, val_pred)
        
        return result
    
    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """XGBoost baseline."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        params = config.get('model', {}) if config else {}
        self.model = xgb.XGBRegressor(
            n_estimators=params.get('xgb_n_estimators', 1000),
            learning_rate=params.get('xgb_learning_rate', 0.01),
            max_depth=params.get('xgb_max_depth', 6),
            subsample=params.get('xgb_subsample', 0.8),
            colsample_bytree=params.get('xgb_colsample_bytree', 0.8),
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        
        # Handle API change: early_stopping_rounds may be in constructor or fit
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=50,
                verbose=False
            )
        except TypeError:
            # Newer API: early_stopping_rounds passed to constructor or callbacks
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
        
        self.is_fitted = True
        
        train_pred = self.model.predict(X_train)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        result = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': r2_score(y_train, train_pred),
        }
        
        # Handle best_iteration_ (newer XGBoost API)
        if hasattr(self.model, 'best_iteration_'):
            result['best_iteration'] = self.model.best_iteration_
        elif hasattr(self.model, 'best_iteration'):
            result['best_iteration'] = self.model.best_iteration
        
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            result['val_mae'] = mean_absolute_error(y_val, val_pred)
            result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            result['val_r2'] = r2_score(y_val, val_pred)
        
        return result
    
    def predict(self, X):
        return self.model.predict(X)


class MLPModel(BaseModel):
    """Multi-layer Perceptron baseline."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        
        params = config.get('model', {}) if config else {}
        self.model = MLPRegressor(
            hidden_layer_sizes=params.get('mlp_hidden_sizes', (100, 50)),
            activation=params.get('mlp_activation', 'relu'),
            solver=params.get('mlp_solver', 'adam'),
            alpha=params.get('mlp_alpha', 0.01),
            max_iter=params.get('mlp_max_iter', 1000),
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        # Scale features for neural network
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
        else:
            X_val_scaled = None
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        train_pred = self.model.predict(X_train_scaled)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        result = {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if X_val_scaled is not None and y_val is not None:
            val_pred = self.model.predict(X_val_scaled)
            result['val_mae'] = mean_absolute_error(y_val, val_pred)
            result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            result['val_r2'] = r2_score(y_val, val_pred)
        
        return result
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        return self.model.predict(X_scaled)


class SVMModel(BaseModel):
    """SVM with RBF kernel baseline."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        from sklearn.svm import SVR
        params = config.get('model', {}) if config else {}
        self.model = SVR(
            kernel='rbf',
            C=params.get('svm_C', 1.0),
            gamma=params.get('svm_gamma', 'scale'),
            epsilon=params.get('svm_epsilon', 0.1)
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        train_pred = self.model.predict(X_train_scaled)
        result = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            result['val_mae'] = mean_absolute_error(y_val, val_pred)
            result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            result['val_r2'] = r2_score(y_val, val_pred)
        
        return result
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        return self.model.predict(X_scaled)


class GaussianProcessModel(BaseModel):
    """Gaussian Process regression baseline."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
        params = config.get('model', {}) if config else {}
        kernel = Matern(length_scale=params.get('gp_length_scale', 1.0), 
                       nu=params.get('gp_nu', 2.5)) + WhiteKernel(
            noise_level=params.get('gp_noise_level', 1e-2))
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=params.get('gp_alpha', 1e-10),
            n_restarts_optimizer=params.get('gp_n_restarts', 5),
            random_state=42
        )
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.scaler_X = StandardScaler()
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        train_pred = self.model.predict(X_train_scaled)
        result = {
            'train_mae': mean_absolute_error(y_train, train_pred),
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
            'train_r2': r2_score(y_train, train_pred)
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            val_pred = self.model.predict(X_val_scaled)
            result['val_mae'] = mean_absolute_error(y_val, val_pred)
            result['val_rmse'] = np.sqrt(mean_squared_error(y_val, val_pred))
            result['val_r2'] = r2_score(y_val, val_pred)
        
        return result
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        return self.model.predict(X_scaled)


class BaselineEnsemble(BaseModel):
    """Simple ensemble of baseline models."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.models = {
            'ridge': RidgeRegressor(config),
            'rf': RandomForestModel(config),
            'xgb': XGBoostModel(config),
        }
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        for name, model in self.models.items():
            model.fit(X_train, y_train, X_val, y_val)
        
        self.is_fitted = True
        return {'train_mae': 0.0}
    
    def predict(self, X):
        predictions = []
        for model in self.models.values():
            predictions.append(model.predict(X))
        
        return np.mean(predictions, axis=0)


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    output_dir: Path = None
) -> Dict[str, Dict[str, float]]:
    """Train all baseline models and evaluate on test set.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        X_test, y_test: Test data
        config: Configuration dictionary
        output_dir: Directory to save results
    
    Returns:
        Dictionary mapping model names to metrics
    """
    results = {}
    
    # Define models to train
    models = {
        'mean': MeanPredictor(strategy='mean'),
        'median': MeanPredictor(strategy='median'),
        'ridge': RidgeRegressor(config),
        'random_forest': RandomForestModel(config),
        'xgboost': XGBoostModel(config),
        'mlp': MLPModel(config),
    }
    
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        
        try:
            # Train
            train_metrics = model.fit(X_train, y_train, X_val, y_val)
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            
            # Compute test metrics
            test_metrics = {
                'test_mae': mean_absolute_error(y_test, y_pred),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'test_r2': r2_score(y_test, y_pred),
                'test_mdae': np.median(np.abs(y_test - y_pred)),
                **train_metrics
            }
            
            results[name] = test_metrics
            
            logger.info(
                f"{name}: Test MAE={test_metrics['test_mae']:.4f}, "
                f"RMSE={test_metrics['test_rmse']:.4f}, "
                f"R²={test_metrics['test_r2']:.4f}"
            )
            
            # Save trained model if output directory specified
            if output_dir:
                import joblib
                output_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, output_dir / f"{name}_model.pkl")
            
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")
            results[name] = {'error': str(e)}
    
    return results


EnsembleModel = BaselineEnsemble

__all__ = [
    'BaseModel',
    'MeanPredictor',
    'RidgeRegressor',
    'RandomForestModel',
    'XGBoostModel',
    'MLPModel',
    'SVMModel',
    'GaussianProcessModel',
    'BaselineEnsemble',
    'EnsembleModel',
    'train_baseline_models',
]
