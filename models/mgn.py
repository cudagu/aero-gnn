import torch
from torch import nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn.functional as F

from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer
    
class MeshGraphNet(nn.Module):
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
                 dropout: float = 0.0,
                 do_concat_trick: bool = False
                 ):
        """
        Args:
            input_node_dim: Dimension of input node features
            input_edge_dim: Dimension of input edge features  
            output_node_dim: Dimension of output predictions
            hidden_dim: Hidden dimension for MLPs
            num_layers: Number of message passing layers
            num_mesh_features: Number of mesh-specific features (positions, etc.)
        """
        super().__init__()
        
        
        # Encoder: project input features to hidden dimension
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

        # Message passing layers
        self.layers = nn.ModuleList([
            MeshGraphNetLayer(node_dim=hidden_dim_processor, 
                             edge_dim=hidden_dim_processor, 
                             hidden_dim=hidden_dim_processor,
                             num_hidden_layers_node_processor=num_hidden_layers_node_processor,
                             num_hidden_layers_edge_processor=num_hidden_layers_edge_processor,
                             activation_fn=activation_fn,
                             use_layer_norm=True,
                             aggregation=aggregation,
                             do_concat_trick=do_concat_trick) 
            for _ in range(processor_size)
        ])
        
        # Decoder: project back to output dimension
        # self.pressureDecoder = MLP(input_dim=hidden_dim_processor,
        #                                     hidden_dim=hidden_dim_decoder,
        #                                     output_dim = 1,
        #                                     num_hidden_layers=num_hidden_layers_decoder,
        #                                     activation_fn=activation_fn,
        #                                     use_layer_norm=False
        #                                     )
        
        # self.shearDecoder = MLP(input_dim=hidden_dim_processor,
        #                                  hidden_dim=hidden_dim_decoder,
        #                                  output_dim = 2,
        #                                  num_hidden_layers=num_hidden_layers_decoder,
        #                                  activation_fn=activation_fn,
        #                                  use_layer_norm=False
        #                                  )
        
        # self.temperatureDecoder = MLP(input_dim=hidden_dim_processor,
        #                                        hidden_dim=hidden_dim_decoder,
        #                                        output_dim = 1,
        #                                        num_hidden_layers=num_hidden_layers_decoder,
        #                                        activation_fn=activation_fn,
        #                                        use_layer_norm=False
        #                                        )
        
        self.decoder = MLP(input_dim=hidden_dim_processor,
                            hidden_dim=hidden_dim_decoder,
                            output_dim = output_node_dim,
                            num_hidden_layers=num_hidden_layers_decoder,
                            activation_fn=activation_fn,
                            use_layer_norm=False)

    def forward(self, 
                node_attr: torch.Tensor,
                edge_attr: torch.Tensor, 
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
        node_hidden = self.node_encoder(node_attr)
        edge_hidden = self.edge_encoder(edge_attr)
        
        # Message passing
        for layer in self.layers:
            node_hidden, edge_hidden = layer(node_hidden, edge_hidden, edge_index)
            
        predictions = self.decoder(node_hidden)
            
        # Decode predictions
        # pressure_predictions = self.pressureDecoder(node_hidden)
        # shear_predictions = self.shearDecoder(node_hidden)
        # temperature_predictions = self.temperatureDecoder(node_hidden)
        
        # predictions = torch.cat([pressure_predictions, shear_predictions, temperature_predictions], dim=-1)
        
        return predictions