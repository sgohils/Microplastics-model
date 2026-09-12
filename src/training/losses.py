"""Loss functions for microplastic prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error loss."""
    
    def __init__(self, weights: Optional[torch.Tensor] = None, 
                 reduction: str = 'mean'):
        super().__init__()
        self.weights = weights
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target) ** 2
        
        if self.weights is not None:
            loss = loss * self.weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QuantileLoss(nn.Module):
    """Quantile loss for quantile regression."""
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions of shape (batch, num_quantiles)
            target: Target values of shape (batch, 1)
        """
        assert pred.shape[-1] == len(self.quantiles)
        assert target.ndim == 2
        
        losses = []
        for i, q in enumerate(self.quantiles):
            pred_q = pred[:, i]
            errors = target.squeeze(-1) - pred_q
            losses.append(torch.max(q * errors, (q - 1) * errors))
        
        loss = torch.stack(losses, dim=-1).mean()
        return loss


class PhysicsInformedLoss(nn.Module):
    """Loss function incorporating physical transport constraints.
    
    This is a placeholder for future implementation.
    Physics-informed constraints are only included when supported
    by established hydrological science (see Stage 1 spec section 31).
    """
    
    def __init__(self, prediction_loss: nn.Module, 
                 transport_weight: float = 0.1):
        super().__init__()
        self.prediction_loss = prediction_loss
        self.transport_weight = transport_weight
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                downstream_pred: Optional[torch.Tensor] = None,
                upstream_pred: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional transport consistency.
        
        Args:
            pred: Predictions
            target: Targets
            downstream_pred: Downstream predictions (optional)
            upstream_pred: Upstream predictions (optional)
        
        Returns:
            Combined loss
        """
        loss = self.prediction_loss(pred, target)
        
        # Only add transport constraint if both available
        if downstream_pred is not None and upstream_pred is not None:
            # Simple mass conservation constraint:
            # downstream concentration should not exceed upstream
            # This is a simplified physical constraint
            transport_loss = F.relu(upstream_pred - downstream_pred).mean()
            loss = loss + self.transport_weight * transport_loss
        
        return loss


def get_loss_function(loss_name: str, device: Optional[torch.device] = None,
                     **kwargs) -> nn.Module:
    """Get loss function by name.
    
    Args:
        loss_name: Name of loss function
        device: Device to place loss function
        **kwargs: Additional arguments
    
    Returns:
        Loss function module
    """
    if loss_name == 'mse':
        loss_fn = nn.MSELoss()
    elif loss_name == 'mae':
        loss_fn = nn.L1Loss()
    elif loss_name == 'huber':
        loss_fn = nn.HuberLoss(delta=kwargs.get('delta', 1.0))
    elif loss_name == 'weighted_mse':
        loss_fn = WeightedMSELoss()
    elif loss_name == 'quantile':
        loss_fn = QuantileLoss(quantiles=kwargs.get('quantiles', [0.1, 0.5, 0.9]))
    elif loss_name == 'smooth_l1':
        loss_fn = nn.SmoothL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
    
    if device is not None:
        loss_fn = loss_fn.to(device)
    
    return loss_fn
