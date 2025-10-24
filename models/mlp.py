import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable layers and activation."""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int, 
                 num_hidden_layers: int = 1, 
                 activation_fn: str = 'relu', 
                 dropout: float = 0.0,
                 use_layer_norm: bool=True):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.use_layer_norm = use_layer_norm
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        # Hidden layers 
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layer
        if num_hidden_layers > 0:
            self.layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            self.layers[-1] = nn.Linear(input_dim, output_dim)
            
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
            
        self.activation = getattr(F, activation_fn)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = self.activation(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x)
        
        if self.use_layer_norm:
            x = self.layer_norm(x)
            
        return x