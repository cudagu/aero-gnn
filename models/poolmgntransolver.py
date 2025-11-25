import torch
from torch import nn
import torch.nn.functional as F

from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer
from models.transolver import TransolverBlock
from einops import rearrange
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class poolMGNTransolver(nn.Module):
    """
    Combines global pooling mechanism from poolMGN with MGNTransolver architecture.
    Uses global pooling to compute graph-level features and broadcasts them to nodes
    before each MGN+Transolver block.
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
                 global_pool_method: str = 'mean',
                 num_hidden_layers_global_encoder: int = 1,
                 global_dim: int = 256,
                 ):

        super().__init__()
        self.__name__ = 'poolMGNTransolver'

        # Global pooling setup
        if global_pool_method == 'mean':
            self.global_pool = global_mean_pool
        elif global_pool_method == 'max':
            self.global_pool = global_max_pool
        elif global_pool_method == 'add':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unsupported global pooling method: {global_pool_method}")

        # Global encoder to process pooled features
        self.global_encoder = MLP(hidden_dim_processor,
                                  hidden_dim=global_dim,
                                  output_dim=global_dim,
                                  num_hidden_layers=num_hidden_layers_global_encoder,
                                  activation_fn=activation_fn,
                                  dropout=dropout,
                                  use_layer_norm=True
                                  )

        # Node encoder: project input features to hidden dimension
        self.node_encoder = MLP(input_node_dim,
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

        # MLP to combine node features with global features
        self.global_fusion = MLP(input_dim=hidden_dim_processor + global_dim,
                                 hidden_dim=hidden_dim_processor,
                                 output_dim=hidden_dim_processor,
                                 num_hidden_layers=1,
                                 activation_fn=activation_fn,
                                 dropout=dropout,
                                 use_layer_norm=True)

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

        # Fuse local and global outputs from each block
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
        # Encode node and edge features
        node_attr = self.node_encoder(node_attr)
        edge_attr = self.edge_encoder(edge_attr)

        for block in self.blocks:
            # Compute global pooling for current node features
            if batch is not None:
                global_features = self.global_encoder(node_attr)
                global_features = self.global_pool(global_features, batch)
                # Broadcast global features to all nodes in each graph
                global_features = global_features[batch]
            else:
                global_features = self.global_encoder(node_attr)
                global_features = self.global_pool(global_features, torch.zeros(node_attr.size(0), dtype=torch.long, device=node_attr.device))
                global_features = global_features.repeat(node_attr.size(0), 1)

            # Combine node features with global features
            node_attr_with_global = torch.cat([node_attr, global_features], dim=-1)
            node_attr_fused = self.global_fusion(node_attr_with_global)

            # Message passing with MeshGraphNet layers
            node_input = node_attr_fused
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
