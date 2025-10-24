"""
BSMS-GNN: Bistride Multi-Scale MeshGraphNet

Implements a multi-scale graph neural network using bistride pooling
for efficient mesh-based physical simulation.

Based on: "Efficient Learning of Mesh-Based Physical Simulation with
Bi-Stride Multi-Scale Graph Neural Network" (ICML 2023)
GitHub: https://github.com/Eydcao/BSMS-GNN
"""

import torch
import torch.nn as nn
from models.mlp import MLP
from models.bistride_ops import BistridePooling, Unpool, WeightedEdgeConv, GMP


class MultiScaleGraphPreprocessor:
    """
    Preprocesses a graph to create multi-scale hierarchy using bistride pooling.

    This should be done once during data loading, not during training.
    """

    def __init__(self, num_levels=3):
        """
        Args:
            num_levels: Number of coarsening levels
        """
        self.num_levels = num_levels

    def create_multiscale_graph(self, data):
        """
        Create multi-scale graph hierarchy.

        Args:
            data: PyTorch Geometric Data object with edge_index, pos

        Returns:
            multi_data: Dictionary containing:
                - 'edge_indices': List of edge_index at each level
                - 'node_indices': List of selected node indices at each level
                - 'num_nodes': List of number of nodes at each level
                - 'positions': List of positions at each level
        """
        device = data.edge_index.device

        multi_data = {
            'edge_indices': [data.edge_index],
            'node_indices': [],
            'num_nodes': [data.pos.shape[0]],
            'positions': [data.pos]
        }

        current_edge_index = data.edge_index
        current_pos = data.pos
        current_num_nodes = data.pos.shape[0]

        for level in range(self.num_levels):
            # Select nodes using bistride pooling
            selected_indices = BistridePooling.select_bistride_nodes(
                current_edge_index,
                current_num_nodes,
                current_pos
            )

            # Store selected indices
            multi_data['node_indices'].append(selected_indices)

            # Create mapping from old to new indices
            index_map = torch.full((current_num_nodes,), -1, dtype=torch.long, device=device)
            index_map[selected_indices] = torch.arange(len(selected_indices), device=device)

            # Coarsen positions
            current_pos = current_pos[selected_indices]
            multi_data['positions'].append(current_pos)

            # Coarsen edges
            src, dst = current_edge_index
            mask = (index_map[src] >= 0) & (index_map[dst] >= 0)
            new_src = index_map[src[mask]]
            new_dst = index_map[dst[mask]]
            current_edge_index = torch.stack([new_src, new_dst], dim=0)

            # Remove self-loops
            mask = current_edge_index[0] != current_edge_index[1]
            current_edge_index = current_edge_index[:, mask]

            multi_data['edge_indices'].append(current_edge_index)
            current_num_nodes = len(selected_indices)
            multi_data['num_nodes'].append(current_num_nodes)

        return multi_data


class BSMSGMP(nn.Module):
    """
    Bistride Multi-Scale Graph Message Passing module.

    U-Net style architecture with encoder (downsampling) and decoder (upsampling) paths.
    """

    def __init__(self,
                 num_levels,
                 latent_dim,
                 hidden_dim,
                 pos_dim=2):
        """
        Args:
            num_levels: Number of coarsening levels
            latent_dim: Dimension of latent features
            hidden_dim: Hidden dimension for MLPs
            pos_dim: Dimension of position features (2 for 2D, 3 for 3D)
        """
        super().__init__()

        self.num_levels = num_levels
        self.latent_dim = latent_dim
        self.pos_dim = pos_dim

        # Downsampling path (encoder)
        self.down_gmps = nn.ModuleList([
            GMP(latent_dim, latent_dim, hidden_dim)
            for _ in range(num_levels + 1)
        ])

        self.down_edge_convs = nn.ModuleList([
            WeightedEdgeConv(latent_dim, latent_dim, aggr='add')
            for _ in range(num_levels)
        ])

        # Bottleneck
        self.bottom_gmp = GMP(latent_dim, latent_dim, hidden_dim)

        # Upsampling path (decoder)
        self.up_edge_convs = nn.ModuleList([
            WeightedEdgeConv(latent_dim, latent_dim, aggr='add')
            for _ in range(num_levels)
        ])

        self.unpools = nn.ModuleList([
            Unpool() for _ in range(num_levels)
        ])

    def forward(self, x, edge_attrs, edge_indices, node_indices, num_nodes_list, positions):
        """
        Forward pass through multi-scale architecture.

        Args:
            x: [num_nodes_finest, latent_dim] - node features at finest level
            edge_attrs: List of [num_edges, latent_dim] - edge features at each level
            edge_indices: List of edge_index at each level
            node_indices: List of selected indices at each level
            num_nodes_list: List of number of nodes at each level
            positions: List of positions at each level

        Returns:
            x: [num_nodes_finest, latent_dim] - processed node features
        """
        skip_connections = []
        edge_weights_down = []

        # Downsampling path (encoder)
        for i in range(self.num_levels):
            # Message passing at current level
            x, edge_attrs[i] = self.down_gmps[i](x, edge_attrs[i], edge_indices[i])

            # Store skip connection
            skip_connections.append(x.clone())

            # Weighted edge convolution
            x_conv, ew = self.down_edge_convs[i](
                x, edge_indices[i], positions[i], compute_weights=True
            )
            edge_weights_down.append(ew)
            x = x + x_conv  # Residual

            # Pool nodes to coarser level
            x = x[node_indices[i]]

        # Bottleneck: Message passing at coarsest level
        x, edge_attrs[-1] = self.bottom_gmp(x, edge_attrs[-1], edge_indices[-1])

        # Upsampling path (decoder)
        for i in range(self.num_levels - 1, -1, -1):
            # Unpool to finer level
            x = self.unpools[i](x, node_indices[i], num_nodes_list[i])

            # Weighted edge convolution (using stored weights)
            x_conv, _ = self.up_edge_convs[i](
                x, edge_indices[i], positions[i],
                edge_weights=edge_weights_down[i],
                compute_weights=False
            )
            x = x + x_conv  # Residual

            # Add skip connection
            x = x + skip_connections[i]

        return x


class BSMS_MeshGraphNet(nn.Module):
    """
    Complete BSMS MeshGraphNet model compatible with the user's setup.
    """

    def __init__(self,
                 input_node_dim: int,
                 input_edge_dim: int,
                 output_node_dim: int,
                 num_levels: int = 3,
                 latent_dim: int = 128,
                 hidden_dim: int = 128,
                 pos_dim: int = 2,
                 num_hidden_layers_encoder: int = 2,
                 num_hidden_layers_decoder: int = 2,
                 activation_fn: str = 'relu',
                 dropout: float = 0.0):
        """
        Args:
            input_node_dim: Dimension of input node features
            input_edge_dim: Dimension of input edge features
            output_node_dim: Dimension of output predictions
            num_levels: Number of coarsening levels
            latent_dim: Dimension of latent features
            hidden_dim: Hidden dimension for MLPs
            pos_dim: Position dimension (2 for 2D, 3 for 3D)
            num_hidden_layers_encoder: Number of hidden layers in encoder
            num_hidden_layers_decoder: Number of hidden layers in decoder
            activation_fn: Activation function name
            dropout: Dropout rate
        """
        super().__init__()

        self.num_levels = num_levels
        self.latent_dim = latent_dim
        self.pos_dim = pos_dim

        # Node encoder
        self.node_encoder = MLP(
            input_dim=input_node_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers_encoder,
            activation_fn=activation_fn,
            dropout=dropout,
            use_layer_norm=True
        )

        # Edge encoder
        self.edge_encoder = MLP(
            input_dim=input_edge_dim,
            hidden_dim=hidden_dim,
            output_dim=latent_dim,
            num_hidden_layers=num_hidden_layers_encoder,
            activation_fn=activation_fn,
            dropout=dropout,
            use_layer_norm=True
        )

        # Multi-scale processor
        self.bsgmp = BSMSGMP(
            num_levels=num_levels,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            pos_dim=pos_dim
        )

        # Decoder
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=output_node_dim,
            num_hidden_layers=num_hidden_layers_decoder,
            activation_fn=activation_fn,
            dropout=dropout,
            use_layer_norm=False
        )

    def forward(self,
                node_attr: torch.Tensor,
                edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                multi_data: dict = None):
        """
        Forward pass.

        Args:
            node_attr: [num_nodes, input_node_dim] - Node features
            edge_attr: [num_edges, input_edge_dim] - Edge features
            edge_index: [2, num_edges] - Edge connectivity
            multi_data: Multi-scale graph data (optional, will be computed if not provided)

        Returns:
            predictions: [num_nodes, output_node_dim]
        """
        # Encode node and edge features
        node_hidden = self.node_encoder(node_attr)
        edge_hidden = self.edge_encoder(edge_attr)

        # If multi_data not provided, create it (not recommended during training)
        if multi_data is None:
            raise ValueError(
                "multi_data must be provided. Use MultiScaleGraphPreprocessor "
                "to preprocess graphs before training."
            )

        # Encode edge features at all levels
        edge_attrs_encoded = [edge_hidden]
        for i in range(1, len(multi_data['edge_indices'])):
            # Pool edge features (simple approach: recompute from pooled nodes)
            # In practice, you might want to precompute these
            coarse_edge_index = multi_data['edge_indices'][i]
            num_edges = coarse_edge_index.shape[1]

            # Initialize with zeros or learn to pool edges
            coarse_edge_attr = torch.zeros(
                num_edges, self.latent_dim,
                device=edge_hidden.device, dtype=edge_hidden.dtype
            )
            edge_attrs_encoded.append(coarse_edge_attr)

        # Multi-scale message passing
        node_hidden = self.bsgmp(
            node_hidden,
            edge_attrs_encoded,
            multi_data['edge_indices'],
            multi_data['node_indices'],
            multi_data['num_nodes'],
            multi_data['positions']
        )

        # Decode predictions
        predictions = self.decoder(node_hidden)

        return predictions


def create_bsms_model_from_config(config):
    """
    Helper function to create BSMS model from configuration dict.

    Args:
        config: Configuration dictionary with model parameters

    Returns:
        model: BSMS_MeshGraphNet instance
    """
    model_config = config.get('model', {})

    return BSMS_MeshGraphNet(
        input_node_dim=model_config.get('input_node_dim'),
        input_edge_dim=model_config.get('input_edge_dim'),
        output_node_dim=model_config.get('output_node_dim'),
        num_levels=model_config.get('num_levels', 3),
        latent_dim=model_config.get('hidden_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 128),
        pos_dim=model_config.get('pos_dim', 2),
        num_hidden_layers_encoder=model_config.get('num_hidden_layers_encoder', 2),
        num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder', 2),
        activation_fn=model_config.get('activation_fn', 'relu'),
        dropout=model_config.get('dropout', 0.0)
    )
