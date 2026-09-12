"""Uncertainty estimation using deep ensembles."""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from torch.utils.data import DataLoader

from ..models.temporal_gnn import SpatioTemporalGNN, GraphModelFactory
from ..training.train_gnn import train_model
from .metrics import compute_metrics
import logging
import json

logger = logging.getLogger(__name__)


class DeepEnsemble:
    """Deep ensemble for uncertainty estimation.
    
    Trains multiple models with different random seeds and initializations
    to estimate predictive uncertainty.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 n_members: int = 5,
                 base_seed: int = 42):
        self.model_config = model_config
        self.n_members = n_members
        self.base_seed = base_seed
        self.models: List[nn.Module] = []
        self.is_fitted = False
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: DataLoader,
            config: Dict[str, Any],
            output_dir: Path,
            device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Train ensemble members.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            output_dir: Directory to save model checkpoints
            device: Compute device
        
        Returns:
            Training results dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for i in range(self.n_members):
            seed = self.base_seed + i * 100
            logger.info(f"Training ensemble member {i+1}/{self.n_members} (seed={seed})")
            
            # Create model
            model = GraphModelFactory.create('st-gnn', self.model_config)
            
            if device is not None:
                model = model.to(device)
            
            # Train model
            checkpoint_path = output_dir / f"ensemble_member_{i}.pt"
            result = train_model(
                model, train_loader, val_loader,
                config, checkpoint_path, seed=seed, device=device
            )
            
            self.models.append(model)
            results[f'member_{i}'] = result
        
        self.is_fitted = True
        
        # Save ensemble metadata
        metadata = {
            'n_members': self.n_members,
            'base_seed': self.base_seed,
            'model_config': self.model_config,
            'trained_at': str(output_dir)
        }
        
        with open(output_dir / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return results
    
    def predict(self, 
                data_loader: DataLoader,
                device: Optional[torch.device] = None,
                return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions with uncertainty estimates.
        
        Args:
            data_loader: Test data loader
            device: Compute device
            return_std: Whether to return standard deviation
        
        Returns:
            Tuple of (mean_prediction, std_prediction)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        all_predictions = []
        
        for member_idx, model in enumerate(self.models):
            model.eval()
            model.to(device)
            
            member_preds = []
            
            with torch.no_grad():
                for batch in data_loader:
                    batch = batch.to(device)
                    predictions = model(batch)
                    
                    if hasattr(batch, 'obs_mask'):
                        predictions = predictions[batch.obs_mask]
                    
                    member_preds.extend(predictions.cpu().numpy())
            
            all_predictions.append(np.array(member_preds))
        
        # Stack predictions: (n_members, n_samples)
        all_predictions = np.stack(all_predictions, axis=0)
        
        # Compute mean and std
        mean_pred = np.mean(all_predictions, axis=0)
        
        if return_std:
            std_pred = np.std(all_predictions, axis=0)
            return mean_pred, std_pred
        
        return mean_pred, None
    
    def predict_with_intervals(self,
                               data_loader: DataLoader,
                               device: Optional[torch.device] = None,
                               confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """Make predictions with confidence intervals.
        
        Args:
            data_loader: Test data loader
            device: Compute device
            confidence: Confidence level (e.g., 0.95 for 95% CI)
        
        Returns:
            Dictionary with prediction, lower, upper, std
        """
        mean_pred, std_pred = self.predict(
            data_loader, device=device, return_std=True
        )
        
        # Compute z-score for confidence level
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        lower = mean_pred - z_score * std_pred
        upper = mean_pred + z_score * std_pred
        
        return {
            'prediction': mean_pred,
            'lower': lower,
            'upper': upper,
            'std': std_pred
        }


class MCDropoutWrapper(nn.Module):
    """Wrapper to enable MC dropout during inference."""
    
    def __init__(self, model: nn.Module, n_samples: int = 10):
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        
        # Enable dropout during inference
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers in the model."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()
    
    def forward(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and std predictions."""
        predictions = []
        
        for _ in range(self.n_samples):
            with torch.no_grad():
                pred = self.model(batch)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred


def compute_coverage(y_true: np.ndarray,
                    lower: np.ndarray,
                    upper: np.ndarray) -> float:
    """Compute prediction interval coverage.
    
    Args:
        y_true: True values
        lower: Lower bounds
        upper: Upper bounds
    
    Returns:
        Coverage proportion (proportion of true values within intervals)
    """
    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def analyze_uncertainty_by_region(
    predictions: Dict[str, np.ndarray],
    region_labels: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Analyze uncertainty across different regions.
    
    Args:
        predictions: Dictionary with 'prediction', 'lower', 'upper', 'std'
        region_labels: Region labels for each sample
    
    Returns:
        Dictionary with uncertainty statistics per region
    """
    std_pred = predictions['std']
    results = {}
    
    for region in np.unique(region_labels):
        mask = region_labels == region
        results[str(region)] = {
            'mean_uncertainty': float(np.mean(std_pred[mask])),
            'std_uncertainty': float(np.std(std_pred[mask])),
            'n_samples': int(np.sum(mask))
        }
    
    return results


def analyze_uncertainty_by_flow_condition(
    predictions: Dict[str, np.ndarray],
    flow_conditions: np.ndarray
) -> Dict[str, Dict[str, float]]:
    """Analyze uncertainty across different flow conditions.
    
    Args:
        predictions: Dictionary with prediction results
        flow_conditions: Flow condition labels (e.g., 'normal', 'high', 'low')
    
    Returns:
        Dictionary with uncertainty statistics per flow condition
    """
    std_pred = predictions['std']
    results = {}
    
    for condition in np.unique(flow_conditions):
        mask = flow_conditions == condition
        results[str(condition)] = {
            'mean_uncertainty': float(np.mean(std_pred[mask])),
            'std_uncertainty': float(np.std(std_pred[mask])),
            'n_samples': int(np.sum(mask))
        }
    
    return results


def evaluate_calibration(
    predictions: Dict[str, np.ndarray],
    confidence_levels: List[float] = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
) -> Dict[str, float]:
    """Evaluate prediction interval calibration.
    
    Args:
        predictions: Dictionary with 'prediction', 'std', and optionally 'y_true'
        confidence_levels: List of confidence levels to evaluate
    
    Returns:
        Dictionary mapping confidence level to calibration error
    """
    from scipy import stats
    
    mean_pred = predictions['prediction']
    std_pred = predictions['std']
    y_true = predictions.get('y_true', None)
    
    calibration_results = {}
    
    for conf in confidence_levels:
        z = stats.norm.ppf((1 + conf) / 2)
        lower = mean_pred - z * std_pred
        upper = mean_pred + z * std_pred
        
        if y_true is not None:
            coverage = compute_coverage(y_true, lower, upper)
            calibration_error = abs(coverage - conf)
            calibration_results[f"coverage_{conf}"] = coverage
            calibration_results[f"calibration_error_{conf}"] = calibration_error
    
    return calibration_results
