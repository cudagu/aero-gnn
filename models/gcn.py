import torch
from torch import nn
import torch.nn.functional as F

from models.mlp import MLP

from models.profiling_utils import get_profiler
from torch_sparse import SparseTensor
    
class GCN(nn.Module):
    """Complete GCN model for mesh-based physical simulations."""

    def __init__(self,
                 input_node_dim: int,
                 output_node_dim: int,
                 processor_size: int = 15,
                 activation_fn: str = 'relu',
                 hidden_dim_processor: int = 128,
                 num_hidden_layers_node_encoder: int = 1,
                 hidden_dim_node_encoder: int = 128,
                 hidden_dim_decoder: int = 128,
                 num_hidden_layers_decoder: int = 1,
                 dropout: float = 0.0,

                 ):
        """
        Args:
            input_node_dim: Dimension of input node features
            output_node_dim: Dimension of output predictions
            hidden_dim: Hidden dimension for MLPs
            num_layers: Number of message passing layers
            num_mesh_features: Number of mesh-specific features (positions, etc.)
        """
        super().__init__()
        
        
        # Encoder: project input features to hidden dimension
        # NOTE: use_layer_norm disabled for 20-25% speedup (profiling showed 23.5% GPU time in LayerNorm)
        self.node_encoder = MLP(input_node_dim,
                                         hidden_dim = hidden_dim_node_encoder,
                                         output_dim = hidden_dim_processor,
                                         num_hidden_layers = num_hidden_layers_node_encoder,
                                         activation_fn=activation_fn,
                                         dropout=dropout,
                                         use_layer_norm=True
                                         )

        # Message passing layers, add processor size gcn layers here.
        from torch_geometric.nn import GCNConv
        self.layers = nn.ModuleList()
        for _ in range(processor_size):
            self.layers.append(GCNConv(hidden_dim_processor, hidden_dim_processor, normalize=True))
        
        
        self.decoder = MLP(input_dim=hidden_dim_processor,
                            hidden_dim=hidden_dim_decoder,
                            output_dim = output_node_dim,
                            num_hidden_layers=num_hidden_layers_decoder,
                            activation_fn=activation_fn,
                            use_layer_norm=False)

    def forward(self, 
                node_attr: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_attr: [num_nodes, input_node_dim] - Node features
            edge_attr: [num_edges, input_edge_dim] - Edge features
            edge_index: [2, num_edges] - Edge connectivity
            mesh_pos: [num_nodes, num_mesh_features] - Mesh positions/features
            
        Returns:
            node_predictions: [num_nodes, output_node_dim]
        """
        # Encode input features

        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=torch.ones(edge_index.size(1)), sparse_sizes=(node_attr.size(0), node_attr.size(0)))

        profiler = get_profiler()
        
        node_hidden = self.node_encoder(node_attr)
        
        # Message passing
        with profiler.profile("gcn_message_passing"):
            for layer in self.layers:
                node_hidden = layer(node_hidden, adj)

        predictions = self.decoder(node_hidden)

        
        return predictions