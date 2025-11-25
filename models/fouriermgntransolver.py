import torch
from torch import nn
import torch.nn.functional as F

from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer
from models.transolver import TransolverBlock
from einops import rearrange
from torch_geometric.utils import to_dense_batch

class FourierMGNTransolver(nn.Module):
    """
    Combines Fourier feature encoding with MGNTransolver architecture.
    Applies Fourier embedding to spatial coordinates before encoding,
    then uses MGN+Transolver blocks for processing.
    """
    def __init__(self,
                 input_node_dim: int,
                 input_edge_dim: int,
                 output_node_dim: int,
                 processor_size: int = 6,
                 message_passing_per_processor: int = 1,
                 activation_fn: str = 'gelu',
                 num_hidden_layers_node_processor: int = 1,
                 num_hidden_layers_edge_processor: int = 1,
                 hidden_dim_processor: int = 256,
                 num_hidden_layers_node_encoder: int = 1,
                 hidden_dim_node_encoder: int = 256,
                 num_hidden_layers_edge_encoder: int = 1,
                 hidden_dim_edge_encoder: int = 256,
                 aggregation: str = 'add',
                 hidden_dim_decoder: int = 256,
                 num_hidden_layers_decoder: int = 1,
                 dropout: float = 0.0,
                 do_concat_trick: bool = False,
                 n_head: int = 8,
                 slice_num: int = 32,
                 act: str = 'gelu',
                 mlp_ratio: int = 4,
                 use_checkpoint: bool = True,
                 fourier_features_dim: int = 2,
                 fourier_freq_start: int = -3,
                 fourier_freq_length: int = 7,
                 ):

        super().__init__()
        self.__name__ = 'FourierMGNTransolver'

        # Fourier embedding parameters
        self.fourier_features_dim = fourier_features_dim
        self.fourier_freq_start = fourier_freq_start
        self.fourier_freq_length = fourier_freq_length

        # Calculate expanded node dimension after Fourier embedding
        # Original features + (cos + sin) * num_frequencies * spatial_dims
        fourier_expansion = 2 * fourier_freq_length * fourier_features_dim
        expanded_node_dim = input_node_dim + fourier_expansion

        # Node encoder: project input features (with Fourier expansion) to hidden dimension
        self.node_encoder = MLP(expanded_node_dim,
                                hidden_dim=hidden_dim_node_encoder,
                                output_dim=hidden_dim_processor,
                                num_hidden_layers=num_hidden_layers_node_encoder,
                                activation_fn=activation_fn,
                                dropout=dropout,
                                use_layer_norm=True
                                )

        self.edge_encoder = MLP(input_edge_dim,
                                hidden_dim=hidden_dim_edge_encoder,
                                output_dim=hidden_dim_processor,
                                num_hidden_layers=num_hidden_layers_edge_encoder,
                                activation_fn=activation_fn,
                                dropout=dropout,
                                use_layer_norm=True
                                )

        self.blocks = nn.ModuleList()

        for i in range(processor_size):
            mgn_layers = nn.ModuleList([MeshGraphNetLayer(
                node_dim=hidden_dim_processor,
                edge_dim=hidden_dim_processor,
                hidden_dim=hidden_dim_processor,
                num_hidden_layers_node_processor=num_hidden_layers_node_processor,
                num_hidden_layers_edge_processor=num_hidden_layers_edge_processor,
                activation_fn=activation_fn,
                use_layer_norm=True,
                aggregation=aggregation,
                do_concat_trick=do_concat_trick) for _ in range(message_passing_per_processor)])

            transolver_block = TransolverBlock(
                num_heads=n_head,
                hidden_dim=hidden_dim_processor,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=output_node_dim,
                slice_num=slice_num,
                last_layer=False,
                use_checkpoint=use_checkpoint
                )

            self.blocks.append(nn.ModuleDict({
                'mgn': mgn_layers,
                'transolver': transolver_block
            }))


        self.fuse = MLP(input_dim=hidden_dim_processor,
                        hidden_dim=hidden_dim_processor,
                        output_dim=hidden_dim_processor,
                        num_hidden_layers=1,
                        activation_fn=activation_fn,
                        use_layer_norm=True)

        # Decoder: project back to output dimension
        self.decoder = MLP(input_dim=hidden_dim_processor,
                            hidden_dim=hidden_dim_decoder,
                            output_dim=output_node_dim,
                            num_hidden_layers=num_hidden_layers_decoder,
                            activation_fn=activation_fn,
                            use_layer_norm=False)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights with truncated normal."""
        def _init(m):
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init)

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

    def forward(self, node_attr, edge_attr, edge_index, batch):
        """
        Args:
            node_attr: [num_nodes, input_node_dim]
            edge_attr: [num_edges, input_edge_dim]
            edge_index: [2, num_edges] - [source, target] node indices
            batch: [num_nodes] - batch indices for each node
        Returns:
            Updated node attributes: [num_nodes, output_node_dim]
        """
        # Apply Fourier embedding to spatial features
        fourier_features = self.fourier_embedding(node_attr)

        # Concatenate Fourier features with original node features
        node_attr_embedded = torch.cat([node_attr, fourier_features], dim=-1)

        # Encode node and edge features
        node_attr = self.node_encoder(node_attr_embedded)
        edge_attr = self.edge_encoder(edge_attr)

        for block in self.blocks:

            node_input = node_attr
            # Message passing with MeshGraphNet layers
            for mgn_layer in block['mgn']:
                local_out, edge_attr = mgn_layer(node_input, edge_attr, edge_index)

            # Prepare for Transolver
            node_attr_dense, mask = to_dense_batch(node_input, batch)
            global_out_dense = block['transolver'](node_attr_dense, mask)
            # Flatten back to original shape
            if batch is not None:
                # Extract only real nodes (not padding)
                global_out = global_out_dense[mask]  # [total_nodes, output_dim]
            else:
                global_out = global_out_dense.squeeze(0)

            # Combine local and global outputs

            node_attr = self.fuse(local_out + global_out)

        # Decode to output features

        return self.decoder(node_attr)
