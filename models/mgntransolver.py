import torch
from torch import nn
import torch.nn.functional as F

from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer
from models.transolver import TransolverBlock
from einops import rearrange
from torch_geometric.utils import to_dense_batch

class MGNTransolver(nn.Module):
    def __init__(self,
                 input_node_dim: int,
                 input_edge_dim: int,
                 output_node_dim: int,
                 processor_size: int = 15,
                 message_passing_per_processor: int = 1,
                 activation_fn: str = 'relu',
                 num_hidden_layers_node_processor: int = 1,
                 num_hidden_layers_edge_processor: int = 1,
                 hidden_dim_processor: int = 128,
                 num_hidden_layers_node_encoder: int = 1,
                 hidden_dim_node_encoder: int = 128,
                 num_hidden_layers_edge_encoder: int = 1,
                 hidden_dim_edge_encoder: int = 128,
                 aggregation: str = 'add',
                 hidden_dim_decoder: int = 128,
                 num_hidden_layers_decoder: int = 1,
                 dropout: float = 0.0,
                 do_concat_trick: bool = False,
                 n_head: int = 8,
                 slice_num: int = 32,
                 act: str = 'gelu',
                 mlp_ratio: int = 4,
                 use_checkpoint: bool = True,
                 ):

        super().__init__()
        self.__name__ = 'MGNTransolver'
        
        
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

        self.edge_encoder = MLP(input_edge_dim,
                                         hidden_dim = hidden_dim_edge_encoder,
                                         output_dim = hidden_dim_processor,
                                         num_hidden_layers = num_hidden_layers_edge_encoder,
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
                use_layer_norm=True,  # Changed from False
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
                last_layer=(i == processor_size - 1),
                use_checkpoint=use_checkpoint
                )
            
            self.blocks.append(nn.ModuleDict({
                'mgn': mgn_layers,
                'transolver': transolver_block
            }))
            
        # Decoder: project back to output dimension
        self.decoder = MLP(input_dim=hidden_dim_processor,
                            hidden_dim=hidden_dim_decoder,
                            output_dim = output_node_dim,
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
            # Message passing with MeshGraphNet layers
            for mgn_layer in block['mgn']:
                node_attr, edge_attr = mgn_layer(node_attr, edge_attr, edge_index)
            
            # Prepare for Transolver
            node_attr_dense, mask = to_dense_batch(node_attr, batch)
            node_attr_dense = block['transolver'](node_attr_dense, mask)
            # Flatten back to original shape
            if batch is not None:
                # Extract only real nodes (not padding)
                node_attr = node_attr_dense[mask]  # [total_nodes, output_dim]
            else:
                node_attr = node_attr_dense.squeeze(0)
        
        # Decode to output features
        # pred = self.decoder(node_attr)
        
        return node_attr