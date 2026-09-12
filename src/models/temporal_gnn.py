"""PyTorch-based GNN models for microplastic prediction."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class TemporalEncoder(nn.Module):
    """GRU-based temporal encoder for sequence processing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            hidden: Initial hidden state
        
        Returns:
            Output tensor of shape (batch, hidden_dim)
        """
        # x: (batch, seq_len, input_dim)
        output, hidden = self.gru(x)
        
        # Return last hidden state
        return hidden[-1]  # (batch, hidden_dim)


class GraphSAGE(nn.Module):
    """GraphSAGE encoder for message passing."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Build layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity())
        
        # Hidden layers
        for i in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim) if hidden_dim > 1 else nn.Identity())
        
        # Output layer
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_dim, output_dim))
            self.bns.append(nn.BatchNorm1d(output_dim) if output_dim > 1 else nn.Identity())
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features (num_nodes, input_dim)
            edge_index: Edge index tensor (2, num_edges)
            edge_attr: Edge features (optional)
        
        Returns:
            Node embeddings (num_nodes, output_dim)
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index, edge_attr) if edge_attr is not None else conv(x, edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        return x


class GCN(nn.Module):
    """Graph Convolutional Network encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x


class GAT(nn.Module):
    """Graph Attention Network encoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        
        self.convs.append(GATConv(input_dim, hidden_dim, heads=heads, 
                                  concat=True, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * heads, hidden_dim, 
                                     heads=heads, concat=True, dropout=dropout))
        
        self.convs.append(GATConv(hidden_dim * heads, output_dim, 
                                 heads=1, concat=False, dropout=dropout))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = self.activation(x)
                x = self.dropout(x)
        
        return x


class SpatioTemporalGNN(nn.Module):
    """Spatiotemporal Graph Neural Network for microplastic prediction.
    
    Architecture (per Stage 1 specification):
    Environmental Features → Feature Encoder → Temporal Encoder → 
    Graph Message Passing → Prediction Head → Microplastic Prediction
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Get dimensions from config
        model_config = config.get('model', {})
        self.input_dim = model_config.get('input_dim', 32)
        self.hidden_dim = model_config.get('hidden_dim', 64)
        self.output_dim = model_config.get('output_dim', 1)
        self.num_gnn_layers = model_config.get('num_gnn_layers', 2)
        self.num_rnn_layers = model_config.get('num_rnn_layers', 1)
        self.dropout = model_config.get('dropout', 0.3)
        self.sequence_length = model_config.get('sequence_length', 7)
        
        # Feature encoder
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout
        )
        
        # Graph neural network
        self.gnn = GraphSAGE(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim // 2,
            num_layers=self.num_gnn_layers,
            dropout=self.dropout
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 4, self.output_dim),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, batch):
        """Forward pass.
        
        Args:
            batch: PyTorch Geometric Batch object
        
        Returns:
            Predictions tensor
        """
        x = batch.x  # (num_nodes, seq_len, input_dim)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, 'edge_attr') else None
        
        batch_size = 1  # Single graph per batch
        num_nodes = batch.num_nodes
        
        # Handle temporal dimension
        if x.dim() == 3 and x.size(1) == self.sequence_length:
            # x is (num_nodes, seq_len, features)
            # Reshape for batch processing: (num_nodes * seq_len, features)
            orig_shape = x.shape
            x_reshaped = x.reshape(-1, self.input_dim)
            
            # Feature encoding
            encoded = self.feature_encoder(x_reshaped)
            
            # Reshape back: (num_nodes, seq_len, hidden_dim)
            encoded = encoded.reshape(num_nodes, self.sequence_length, self.hidden_dim)
            
            # Temporal encoding
            temporal_out = self.temporal_encoder(encoded)  # (num_nodes, hidden_dim)
        else:
            # No temporal dimension
            encoded = self.feature_encoder(x)
            temporal_out = encoded
        
        # Graph message passing
        node_emb = self.gnn(temporal_out, edge_index, edge_attr)
        
        # Prediction head
        predictions = self.prediction_head(node_emb)
        
        return predictions.squeeze(-1)


class SpatioTemporalGNNWithUncertainty(SpatioTemporalGNN):
    """Spatiotemporal GNN with built-in uncertainty estimation.
    
    Uses Monte Carlo dropout for uncertainty estimation during inference.
    """
    
    def __init__(self, config: Dict[str, Any], n_samples: int = 10):
        super().__init__(config)
        self.n_samples = n_samples
        # Keep dropout active during inference for MC dropout
        self.mc_dropout = config.get('model', {}).get('mc_dropout', False)
    
    def enable_mc_dropout(self):
        """Enable Monte Carlo dropout for uncertainty estimation."""
        self.mc_dropout = True
    
    def forward(self, batch, return_uncertainty: bool = False):
        if return_uncertainty:
            # MC dropout inference
            predictions = []
            for _ in range(self.n_samples):
                pred = super().forward(batch)
                predictions.append(pred)
            
            predictions = torch.stack(predictions)
            mean_pred = predictions.mean(dim=0)
            std_pred = predictions.std(dim=0)
            
            return mean_pred, std_pred
        else:
            return super().forward(batch)


class GraphModelFactory:
    """Factory for creating different graph model types."""
    
    @staticmethod
    def create(model_type: str, config: Dict[str, Any]) -> nn.Module:
        """Create a model of the specified type.
        
        Args:
            model_type: Type of model ('gnn', 'gcn', 'gat', 'st-gnn')
            config: Configuration dictionary
        
        Returns:
            PyTorch model
        """
        if model_type == 'st-gnn':
            return SpatioTemporalGNN(config)
        elif model_type == 'gcn':
            return GCN(
                input_dim=config.get('model', {}).get('input_dim', 32),
                hidden_dim=config.get('model', {}).get('hidden_dim', 64),
                output_dim=config.get('model', {}).get('output_dim', 1),
                num_layers=config.get('model', {}).get('num_gnn_layers', 2),
                dropout=config.get('model', {}).get('dropout', 0.3)
            )
        elif model_type == 'gat':
            return GAT(
                input_dim=config.get('model', {}).get('input_dim', 32),
                hidden_dim=config.get('model', {}).get('hidden_dim', 64),
                output_dim=config.get('model', {}).get('output_dim', 1),
                num_layers=config.get('model', {}).get('num_gnn_layers', 2),
                dropout=config.get('model', {}).get('dropout', 0.3)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
