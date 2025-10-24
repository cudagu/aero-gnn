import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn.functional as F

from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer


class FourierMeshGraphNet(nn.Module):
    """MeshGraphNet with Fourier embeddings applied to node features before encoding.
    
    The Fourier embedding adds frequency-based features to capture periodic patterns
    in the spatial coordinates, which can help the network better represent fine-grained
    geometric details.
    """
    
    def __init__(self, 
                 input_node_dim: int,
                 input_edge_dim: int,
                 output_node_dim: int,
                 processor_size: int = 15,
                 activation_fn: str = 'relu',
                 num_hidden_layers_node_processor: int = 1,
                 num_hidden_layers_edge_processor: int = 1,
                 hidden_dim_processor: int = 128,
                 num_hidden_layers_node_encoder: int = 1,
                 hidden_dim_node_encoder: int = 128,
                 num_hidden_layers_edge_encoder: int = 1,
                 hidden_dim_edge_encoder: int = 128,
                 aggregation: str = 'sum',
                 hidden_dim_decoder: int = 128,
                 num_hidden_layers_decoder: int = 1,
                 dropout: float = 0.0,
                 fourier_features_dim: int = 2,
                 fourier_freq_start: int = -3,
                 fourier_freq_length: int = 7,
                 ):
        """
        Args:
            input_node_dim: Dimension of input node features
            input_edge_dim: Dimension of input edge features  
            output_node_dim: Dimension of output predictions
            processor_size: Number of message passing layers
            activation_fn: Activation function ('relu', 'gelu', 'silu', etc.)
            num_hidden_layers_node_processor: Hidden layers in node processor MLP
            num_hidden_layers_edge_processor: Hidden layers in edge processor MLP
            hidden_dim_processor: Hidden dimension for processor
            num_hidden_layers_node_encoder: Hidden layers in node encoder MLP
            hidden_dim_node_encoder: Hidden dimension for node encoder
            num_hidden_layers_edge_encoder: Hidden layers in edge encoder MLP
            hidden_dim_edge_encoder: Hidden dimension for edge encoder
            aggregation: Message aggregation method ('sum', 'mean', 'max')
            hidden_dim_decoder: Hidden dimension for decoder
            num_hidden_layers_decoder: Hidden layers in decoder MLP
            dropout: Dropout rate
            fourier_features_dim: Number of spatial dimensions to apply Fourier embedding (e.g., 2 for x,y or 3 for x,y,z)
            fourier_freq_start: Starting frequency index (e.g., -3 means 2^-3)
            fourier_freq_length: Number of frequencies to use
        """
        super().__init__()
        
        self.fourier_features_dim = fourier_features_dim
        self.fourier_freq_start = fourier_freq_start
        self.fourier_freq_length = fourier_freq_length
        
        # Calculate expanded node dimension after Fourier embedding
        # Original features + (cos + sin) * num_frequencies * spatial_dims
        fourier_expansion = 2 * fourier_freq_length * fourier_features_dim
        expanded_node_dim = input_node_dim + fourier_expansion
        
        # Encoder: project input features to hidden dimension
        # Node encoder now takes expanded dimension due to Fourier features
        self.node_encoder = MLP(expanded_node_dim, 
                                hidden_dim=hidden_dim_node_encoder, 
                                output_dim=hidden_dim_processor,
                                num_hidden_layers=num_hidden_layers_node_encoder,
                                activation_fn=activation_fn,
                                dropout=dropout,
                                use_layer_norm=True)
        
        self.edge_encoder = MLP(input_edge_dim, 
                                hidden_dim=hidden_dim_edge_encoder, 
                                output_dim=hidden_dim_processor,
                                num_hidden_layers=num_hidden_layers_edge_encoder,
                                activation_fn=activation_fn,
                                dropout=dropout,
                                use_layer_norm=True)

        # Message passing layers
        self.layers = nn.ModuleList([
            MeshGraphNetLayer(node_dim=hidden_dim_processor, 
                             edge_dim=hidden_dim_processor, 
                             hidden_dim=hidden_dim_processor,
                             num_hidden_layers_node_processor=num_hidden_layers_node_processor,
                             num_hidden_layers_edge_processor=num_hidden_layers_edge_processor,
                             activation_fn=activation_fn,
                             use_layer_norm=True,
                             aggregation=aggregation) 
            for _ in range(processor_size)
        ])
        
        # Decoder: project back to output dimension
        self.decoder = MLP(input_dim=hidden_dim_processor,
                          hidden_dim=hidden_dim_decoder,
                          output_dim=output_node_dim,
                          num_hidden_layers=num_hidden_layers_decoder,
                          activation_fn=activation_fn,
                          use_layer_norm=False)

    def fourier_embedding(self, pos: torch.Tensor) -> torch.Tensor:
        """Apply Fourier feature embedding to spatial coordinates.
        
        Computes: [cos(2^i * pi * x), sin(2^i * pi * x)] for i in [freq_start, freq_start+freq_length)
        
        Args:
            pos: [num_nodes, spatial_dim] - Spatial coordinates (first fourier_features_dim features)
            
        Returns:
            embedding: [num_nodes, 2*freq_length*spatial_dim] - Fourier embedded features
        """
        # Extract spatial features (first fourier_features_dim dimensions)
        spatial_features = pos[:, :self.fourier_features_dim]
        
        # Generate frequency indices
        freq_indices = torch.arange(
            self.fourier_freq_start, 
            self.fourier_freq_start + self.fourier_freq_length,
            device=pos.device,
            dtype=pos.dtype
        )
        
        # Compute frequencies: 2^i * pi
        frequencies = (2.0 ** freq_indices) * torch.pi
        
        # Expand dimensions for broadcasting: [1, 1, num_freqs]
        frequencies = frequencies.view(1, 1, -1)
        
        # Expand spatial features: [num_nodes, spatial_dim, 1]
        spatial_expanded = spatial_features.unsqueeze(-1)
        
        # Compute cosine and sine features
        cos_features = torch.cos(frequencies * spatial_expanded)  # [num_nodes, spatial_dim, num_freqs]
        sin_features = torch.sin(frequencies * spatial_expanded)  # [num_nodes, spatial_dim, num_freqs]
        
        # Concatenate cos and sin, then flatten
        # Shape: [num_nodes, spatial_dim, 2*num_freqs] -> [num_nodes, 2*spatial_dim*num_freqs]
        fourier_features = torch.cat([cos_features, sin_features], dim=-1)
        fourier_features = fourier_features.reshape(pos.shape[0], -1)
        
        return fourier_features

    def forward(self, 
                node_attr: torch.Tensor,
                edge_attr: torch.Tensor, 
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_attr: [num_nodes, input_node_dim] - Node features
            edge_attr: [num_edges, input_edge_dim] - Edge features
            edge_index: [2, num_edges] - Edge connectivity
            
        Returns:
            node_predictions: [num_nodes, output_node_dim]
        """
        # Apply Fourier embedding to spatial features
        fourier_features = self.fourier_embedding(node_attr)
        
        # Concatenate Fourier features with original node features
        node_attr_embedded = torch.cat([node_attr, fourier_features], dim=-1)
        
        # Encode input features
        node_hidden = self.node_encoder(node_attr_embedded)
        edge_hidden = self.edge_encoder(edge_attr)
        
        # Message passing
        for layer in self.layers:
            node_hidden, edge_hidden = layer(node_hidden, edge_hidden, edge_index)
            
        # Decode predictions
        predictions = self.decoder(node_hidden)
        
        return predictions