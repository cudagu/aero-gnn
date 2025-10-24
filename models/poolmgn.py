import torch
from torch import nn
from torch.nn import Linear, Sequential
from torch_scatter import scatter_mean, scatter_add

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn.functional as F
from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer

class poolMGN(nn.Module):
    """Complete MeshGraphNet model for mesh-based physical simulations."""
    
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
                 global_pool_method: str = 'mean',
                 num_hidden_layers_global_encoder: int = 1,
                 global_dim: int = 128,
                 dropout: float = 0.0,
                 ):
        super().__init__()
        
        if global_pool_method == 'mean':
            self.global_pool = global_mean_pool
        elif global_pool_method == 'max':
            self.global_pool = global_max_pool
        elif global_pool_method == 'add':
            self.global_pool = global_add_pool
        else:
            raise ValueError(f"Unsupported global pooling method: {global_pool_method}")
        
        self.node_encoder = MLP(input_node_dim + global_dim,
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

        self.global_encoder = MLP(input_node_dim,
                                  hidden_dim=global_dim, 
                                  output_dim=global_dim, 
                                  num_hidden_layers=num_hidden_layers_global_encoder,
                                  activation_fn=activation_fn,
                                  dropout=dropout,
                                  use_layer_norm=False
                                  )
        
        self.layers = nn.ModuleList([MeshGraphNetLayer(node_dim=hidden_dim_processor, 
                                                      edge_dim=hidden_dim_processor, 
                                                      hidden_dim=hidden_dim_processor,
                                                      num_hidden_layers_node_processor=num_hidden_layers_node_processor,
                                                      num_hidden_layers_edge_processor=num_hidden_layers_edge_processor,
                                                      activation_fn=activation_fn,
                                                      use_layer_norm=True,
                                                      aggregation=aggregation) 
                                     for _ in range(processor_size)])
        
        # self.pressureDecoder = MLP(input_dim=hidden_dim_processor,
        #                            hidden_dim=hidden_dim_decoder,
        #                            output_dim=1,
        #                            num_hidden_layers=num_hidden_layers_decoder,
        #                            activation_fn=activation_fn,
        #                            use_layer_norm=False
        #                         )
        
        # self.shearDecoder = MLP(input_dim=hidden_dim_processor,
        #                         hidden_dim=hidden_dim_decoder,
        #                         output_dim=2,
        #                         num_hidden_layers=num_hidden_layers_decoder,
        #                         activation_fn=activation_fn,
        #                         use_layer_norm=False
        #                         )   
        
        # self.temperatureDecoder = MLP(input_dim=hidden_dim_processor,
        #                               hidden_dim=hidden_dim_decoder,
        #                               output_dim=1,
        #                               num_hidden_layers=num_hidden_layers_decoder,
        #                               activation_fn=activation_fn,
        #                               use_layer_norm=False
        #                               )
        
        self.decoder = MLP(input_dim=hidden_dim_processor,
                           hidden_dim=hidden_dim_decoder,
                           output_dim=output_node_dim,
                           num_hidden_layers=num_hidden_layers_decoder,
                           activation_fn=activation_fn,
                           use_layer_norm=False
                           )
        
    def forward(self,
                node_attr: torch.Tensor,
                edge_attr: torch.Tensor, 
                edge_index: torch.Tensor, 
                batch: torch.Tensor = None):
        """
        Args:
            node_attr: [num_nodes, input_node_dim]
            edge_attr: [num_edges, input_edge_dim]
            edge_index: [2, num_edges] - [source, target] node indices
            batch: [num_nodes] - batch indices for each node (optional)
        Returns:
            pressure: [num_nodes, 1]
            shear: [num_nodes, 2]
            temperature: [num_nodes, 1]
        """
        
        if batch is not None:
            global_features = self.global_encoder(node_attr)
            global_features = self.global_pool(global_features, batch)
            global_features = global_features.repeat_interleave(torch.bincount(batch), dim=0)
        else:
            global_features = self.global_encoder(node_attr)
            global_features = self.global_pool(global_features, torch.zeros(node_attr.size(0), dtype=torch.long, device=node_attr.device))
            global_features = global_features.repeat(node_attr.size(0), 1)
        
        x = torch.cat((node_attr, global_features), dim=-1)
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for layer in self.layers:
            x, edge_attr = layer(x, edge_attr, edge_index)
        
        # pressure = self.pressureDecoder(x)
        # shear = self.shearDecoder(x)
        # temperature = self.temperatureDecoder(x)

        # predictions = torch.cat([pressure, shear, temperature], dim=-1)
        
        predictions = self.decoder(x)

        return predictions
