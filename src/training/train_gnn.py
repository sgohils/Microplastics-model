"""Training loop and checkpointing."""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import json
from datetime import datetime

from ..utils.seed import set_seed
from .losses import get_loss_function

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(self, patience: int = 30, min_delta: float = 0.0001,
                 mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif self.mode == 'min':
            if val_score < self.best_score - self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'max':
            if val_score > self.best_score + self.min_delta:
                self.best_score = val_score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop
    
    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False


class ModelCheckpoint:
    """Model checkpointing callback."""
    
    def __init__(self, path: Path, mode: str = 'min'):
        self.path = Path(path)
        self.mode = mode
        self.best_score = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
    
    def __call__(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                 epoch: int, val_score: float, 
                 scaler_state: Optional[Dict] = None,
                 additional_state: Optional[Dict] = None):
        """Save model checkpoint.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            val_score: Validation metric
            scaler_state: Optional scaler state dict
            additional_state: Any additional state to save
        """
        should_save = False
        
        if self.best_score is None:
            should_save = True
        elif self.mode == 'min' and val_score < self.best_score:
            should_save = True
        elif self.mode == 'max' and val_score > self.best_score:
            should_save = True
        
        if should_save:
            self.best_score = val_score
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_score': val_score,
                'timestamp': datetime.now().isoformat(),
            }
            
            if scaler_state:
                checkpoint['scaler_state'] = scaler_state
            if additional_state:
                checkpoint.update(additional_state)
            
            torch.save(checkpoint, self.path)
            logger.info(f"Saved best model checkpoint to {self.path}")
        
        return should_save


def train_gnn_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_norm: float = 5.0
) -> Tuple[float, float]:
    """Train model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Compute device
        max_norm: Gradient clipping norm
    
    Returns:
        Tuple of (average loss, MAE)
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    for batch in train_loader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch)
        
        # Compute loss only on nodes with observations
        if hasattr(batch, 'obs_mask'):
            loss = criterion(predictions[batch.obs_mask], 
                           batch.y[batch.obs_mask])
        else:
            loss = criterion(predictions, batch.y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Compute MAE
        with torch.no_grad():
            if hasattr(batch, 'obs_mask'):
                mae = torch.mean(torch.abs(
                    predictions[batch.obs_mask] - batch.y[batch.obs_mask]
                ))
            else:
                mae = torch.mean(torch.abs(predictions - batch.y))
            total_mae += mae.item()
    
    return total_loss / max(1, num_batches), total_mae / max(1, num_batches)


def evaluate_gnn(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate model on data.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        criterion: Loss function
        device: Compute device
    
    Returns:
        Tuple of (average loss, MAE)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            predictions = model(batch)
            
            if hasattr(batch, 'obs_mask'):
                loss = criterion(predictions[batch.obs_mask], 
                               batch.y[batch.obs_mask])
                batch_mae = torch.mean(torch.abs(
                    predictions[batch.obs_mask] - batch.y[batch.obs_mask]
                ))
                all_preds.extend(predictions[batch.obs_mask].cpu().numpy())
                all_targets.extend(batch.y[batch.obs_mask].cpu().numpy())
            else:
                loss = criterion(predictions, batch.y)
                batch_mae = torch.mean(torch.abs(predictions - batch.y))
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(batch.y.cpu().numpy())
            
            total_loss += loss.item()
            total_mae += batch_mae.item()
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    avg_mae = total_mae / max(1, num_batches)
    
    return avg_loss, avg_mae, all_preds, all_targets


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    checkpoint_path: Optional[Path] = None,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Complete training loop with checkpointing and early stopping.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        checkpoint_path: Path to save checkpoint
        seed: Random seed
        device: Compute device
    
    Returns:
        Training results dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    set_seed(seed)
    
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    general_config = config.get('general', {})
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.get('learning_rate', 0.001),
        weight_decay=training_config.get('weight_decay', 0.0001)
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5
    )
    
    # Setup loss
    criterion = get_loss_function(
        training_config.get('loss_function', 'mse'),
        device=device
    )
    
    # Setup early stopping
    early_stopping = EarlyStopping(
        patience=training_config.get('early_stopping_patience', 30),
        mode='min'
    )
    
    # Setup checkpointing
    checkpoint = None
    if checkpoint_path:
        checkpoint = ModelCheckpoint(checkpoint_path, mode='min')
    
    # Training loop
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 
               'val_mae': [], 'lr': []}
    
    max_epochs = training_config.get('max_epochs', 500)
    
    for epoch in range(max_epochs):
        # Train
        train_loss, train_mae = train_gnn_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validate
        val_loss, val_mae, _, _ = evaluate_gnn(
            model, val_loader, criterion, device
        )
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Checkpoint
        if checkpoint:
            checkpoint(model, optimizer, epoch, val_loss,
                      scaler_state=None,
                      additional_state={
                          'config': config,
                          'seed': seed,
                          'epoch': epoch
                      })
        
        # Early stopping
        if early_stopping(val_loss):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Train MAE: {train_mae:.4f} | "
                f"Val MAE: {val_mae:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f}"
            )
    
    # Save final history
    if checkpoint_path:
        history_path = checkpoint_path.parent / f"{checkpoint_path.stem}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    results = {
        'final_train_loss': train_loss,
        'final_train_mae': train_mae,
        'final_val_loss': val_loss,
        'final_val_mae': val_mae,
        'best_val_loss': early_stopping.best_score,
        'epochs_trained': epoch + 1,
        'history': history
    }
    
    return results


def load_checkpoint(model: nn.Module, checkpoint_path: Path,
                   device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto
    
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint
