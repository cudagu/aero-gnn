import torch
from torch import nn
from torch.nn import Linear, Sequential
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter_mean, scatter
from torch_geometric.nn import GCNConv, GINEConv, GraphSAGE
import torch_geometric
import torch.nn.functional as F

def build_mlp(input_dim, hidden_dim, output_dim, num_hidden_layers=2, lay_norm=True, pooling=False, dropout=0.0):
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_hidden_layers):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    layers.append(nn.Linear(hidden_dim, output_dim))
    if lay_norm: return nn.Sequential(*layers, nn.LayerNorm(output_dim))
    return nn.Sequential(*layers)


class NodeEncoder(nn.Module):
    def __init__(self, node_input_size, hidden_channels, num_hidden_layers=2):
        super(NodeEncoder, self).__init__()
        self.mlp = build_mlp(node_input_size, hidden_channels, hidden_channels, num_hidden_layers)

    def forward(self, x):
        return self.mlp(x)
    
class EdgeEncoder(nn.Module):
    def __init__(self, edge_input_size, hidden_channels , num_hidden_layers=2):
        super(EdgeEncoder, self).__init__()
        self.mlp = build_mlp(edge_input_size, hidden_channels, hidden_channels, num_hidden_layers)

    def forward(self, edge_attr):
        return self.mlp(edge_attr)
    
    
class GlobalEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_encoder_layers=2):
        super(GlobalEncoder, self).__init__()
        self.mlp = build_mlp(in_channels, hidden_channels, hidden_channels, num_hidden_layers=num_encoder_layers, lay_norm=False, dropout=0.0)
        self.pool = global_mean_pool  
        self.linout = torch.nn.Linear(hidden_channels,hidden_channels)          
                

    def forward(self, x, edge_index=None,batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0),dtype=torch.long,device=x.device)
        
        X = self.mlp(x)
        X = self.linout(X)
        X = self.pool(X,batch)  # shape: [batch_size, out_channels]
        return X  # shape: [N, out_channels] or [batch_size, out_channels]
    

    
class MeshGraphNetLayer_v2(nn.Module):
    def __init__(self, hidden_channels):
        super(MeshGraphNetLayer_v2, self).__init__()
        self.edge_mlp = build_mlp(hidden_channels, hidden_channels, hidden_channels)
        self.node_mlp = build_mlp(2 * hidden_channels, hidden_channels, hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        e_in = edge_attr
        e_upd = self.edge_mlp(e_in)
        edge_attr = edge_attr + e_upd
        
        
        agg = scatter_mean(edge_attr, col, dim=0, dim_size=x.size(0))
        n_in = torch.cat([x, agg], dim=-1)
        n_upd = self.node_mlp(n_in)
        x = x + n_upd
        
        return x, edge_attr
    
class MeshGraphNet_v2(nn.Module):
    def __init__(self, node_input_size, edge_input_size, hidden_channels, out_channels, num_graph_conv_layers, num_encoder_layers=2, num_decoder_layers=2, dropout=0.0):
        super(MeshGraphNet_v2, self).__init__()
        self.node_encoder = NodeEncoder(node_input_size+hidden_channels, hidden_channels, num_hidden_layers=num_encoder_layers)
        self.edge_encoder = EdgeEncoder(edge_input_size, hidden_channels, num_hidden_layers=num_encoder_layers)
        self.extract_feature = GlobalEncoder(node_input_size, hidden_channels, num_encoder_layers=num_encoder_layers)
        
        self.layers = nn.ModuleList(
            [MeshGraphNetLayer_v2(hidden_channels) for _ in range(num_graph_conv_layers)]
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.decoder = build_mlp(input_dim = hidden_channels,
                                 hidden_dim = hidden_channels,
                                 output_dim=out_channels,
                                 num_hidden_layers=num_decoder_layers - 1,
                                 lay_norm=False,
                                 dropout=dropout)

    def forward(self, node_attr: torch.Tensor,
                edge_attr: torch.Tensor, 
                edge_index: torch.Tensor, batch ):
        
        if batch is not None:
            
            global_features = self.extract_feature(node_attr, edge_index, batch)
            global_features = global_features.repeat_interleave(torch.bincount(batch),dim = 0)
            
        else: 
            global_features = self.extract_feature(node_attr, edge_index, batch)
            global_features = global_features.repeat(node_attr.size(0),1)
        
        x = torch.cat((node_attr, global_features), dim=-1)
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)
        
        return self.decoder(x)
    

