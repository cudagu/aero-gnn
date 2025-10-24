import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn.functional as F

from models.mlp import MLP

class MLPNet(nn.Module):
    """Multi-Layer Perceptron Network for node-level predictions."""
    
    def __init__(self, 
                 input_node_dim: int,
                 output_node_dim: int,
                 hidden_dim: int = 128,
                 num_hidden_layers_encoder: int = 2,
                 num_hidden_layers_decoder: int = 2,
                 activation_fn: str = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        
        self.mlp = MLP(input_dim=input_node_dim,
                       hidden_dim=hidden_dim,
                       output_dim=hidden_dim,
                       num_hidden_layers=num_hidden_layers_encoder,
                       activation_fn=activation_fn,
                       dropout=dropout,
                       use_layer_norm=True)
        
        self.decoder = MLP(input_dim=hidden_dim,
                           hidden_dim=hidden_dim,
                           output_dim=output_node_dim,
                           num_hidden_layers=num_hidden_layers_decoder,
                           activation_fn=activation_fn,
                           dropout=dropout,
                           use_layer_norm=True)

    def forward(self, node_attr):

        node_embedding = self.mlp(node_attr)

        predictions = self.decoder(node_embedding)
        return predictions