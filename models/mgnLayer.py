import torch
from torch import nn
from torch.nn import Linear, Sequential
from torch_scatter import scatter_mean, scatter_add
import torch_geometric
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
import torch.nn.functional as F
from models.mlp import MLP

class EdgeBlock(nn.Module):
    """Edge processing block for MeshGraphNet."""
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 1,
                 activation_fn: str = 'relu',
                 use_layer_norm: bool=True,
                 ):
        super().__init__()
        # Edge features + sender node + receiver node
        input_dim = edge_dim + 2 * node_dim
        # input_dim = edge_dim
        self.mlp = MLP(input_dim = input_dim, 
                        hidden_dim = hidden_dim, 
                        output_dim = edge_dim, 
                        num_hidden_layers = num_hidden_layers,
                        activation_fn=activation_fn,
                        use_layer_norm=use_layer_norm)

    def forward(self, edge_attr, node_attr, edge_index):
        """
        Args:
            edge_attr: [num_edges, edge_dim]
            node_attr: [num_nodes, node_dim] 
            edge_index: [2, num_edges] - [source, target] node indices
        """
        row, col = edge_index 
        sender_nodes = node_attr[row]  # [num_edges, node_dim]
        receiver_nodes = node_attr[col]  # [num_edges, node_dim]
        
        # Concatenate edge features with sender and receiver node features
        edge_input = torch.cat([edge_attr, sender_nodes, receiver_nodes], dim=-1)
        # edge_input = edge_attr
        
        edge_update = self.mlp(edge_input)
        
        return edge_update
    
class EdgeBlockSum(nn.Module):
    """
    Sum trick + pre-project nodes to avoid [E, node_dim] gathers.

      h0 = W_e * e
         + (W_s * x)[row]
         + (W_d * x)[col]
         + b
    """
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 1,
                 activation_fn: str = 'relu',
                 use_layer_norm: bool = True):
        super().__init__()
        self.edge_dim = edge_dim
        self.src_dim = node_dim
        self.dst_dim = node_dim
        
        tmp_lin = nn.Linear(self.edge_dim + self.src_dim + self.dst_dim, hidden_dim, bias=True)
        orig_weight = tmp_lin.weight.data
        w_e, w_s, w_d = torch.split(orig_weight, [self.edge_dim, self.src_dim, self.dst_dim], dim=1)
        
        self.edge_lin = nn.Parameter(w_e)
        self.src_lin = nn.Parameter(w_s)
        self.dst_lin = nn.Parameter(w_d)
        self.bias = tmp_lin.bias
        
        activation_fn = nn.ReLU()
        layers = [activation_fn]
        self.num_hidden_layers = num_hidden_layers
        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, edge_dim))
        
        if use_layer_norm:
            layers.append(nn.LayerNorm(edge_dim))
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, edge_attr, node_attr, edge_index):
        src_feat = node_attr
        dst_feat = node_attr
        
        mlp_edge_attr = F.linear(edge_attr, self.edge_lin, None)
        mlp_src_feat = F.linear(src_feat, self.src_lin, None)
        mlp_dst_feat = F.linear(dst_feat, self.dst_lin, self.bias)
        
        src, dst = edge_index.long()
        
        mlp_sum = mlp_edge_attr + mlp_src_feat[src] + mlp_dst_feat[dst]
        edge_update = self.mlp(mlp_sum)
        return edge_update
        
        
        

    
class NodeBlock(nn.Module):
    """Node processing block for MeshGraphNet."""
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 hidden_dim: int = 128,
                 num_hidden_layers: int = 1,
                 activation_fn: str = 'relu',
                 use_layer_norm: bool=True,
                 aggregation: str = 'add'
                 ):
        super().__init__()
        # Node features + aggregated edge features
        input_dim = node_dim + edge_dim
        self.aggregation = aggregation
        self.mlp = MLP(input_dim=input_dim, 
                        hidden_dim=hidden_dim, 
                        output_dim=node_dim, 
                        num_hidden_layers=num_hidden_layers,
                        activation_fn=activation_fn,
                        use_layer_norm=use_layer_norm)

    def forward(self, node_attr, edge_attr, edge_index):
        """
        Args:
            node_attr: [num_nodes, node_dim]
            edge_attr: [num_edges, edge_dim]
            edge_index: [2, num_edges] - [source, target] node indices
        """
        row, col = edge_index 
        # Aggregate edge features for each node (using sum aggregation)
        if self.aggregation == 'mean':
            edge_aggr = scatter_mean(edge_attr, col, dim=0, dim_size=node_attr.size(0))
        elif self.aggregation == 'add':  
            edge_aggr = scatter_add(edge_attr, col, dim=0, dim_size=node_attr.size(0))
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation}")
        
        # # Concatenate node features with aggregated edge features
        node_input = torch.cat([node_attr, edge_aggr], dim=-1)
        
        return self.mlp(node_input)
    
    
class MeshGraphNetLayer(nn.Module):
    """Single layer of MeshGraphNet with edge and node processing blocks."""
    
    def __init__(self, 
                 node_dim: int, 
                 edge_dim: int, 
                 hidden_dim: int = 128,
                 num_hidden_layers_node_processor: int = 1,
                 num_hidden_layers_edge_processor: int = 1,
                 activation_fn: str = 'relu',
                 use_layer_norm: bool=True,
                 aggregation: str = 'add',
                 do_concat_trick: bool = False
                 ):
        super().__init__()
        if do_concat_trick:
            self.edge_block = EdgeBlockSum(node_dim, edge_dim, hidden_dim, num_hidden_layers_edge_processor, activation_fn, use_layer_norm)
        else:
            self.edge_block = EdgeBlock(node_dim, edge_dim, hidden_dim, num_hidden_layers_edge_processor, activation_fn, use_layer_norm)
        self.node_block = NodeBlock(node_dim, edge_dim, hidden_dim, num_hidden_layers_node_processor, activation_fn, use_layer_norm, aggregation)

    def forward(self, node_attr, edge_attr, edge_index):
        """
        Args:
            node_attr: [num_nodes, node_dim]
            edge_attr: [num_edges, edge_dim]  
            edge_index: [2, num_edges]
        """
        # Process edges
        # Memory monitoring - before edge_block
        mem_before = max_mem_before = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            max_mem_before = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        edge_attr_new = self.edge_block(edge_attr, node_attr, edge_index)
        
        # Memory monitoring - after edge_block
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            max_mem_after = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
            mem_diff = mem_after - mem_before
            max_mem_diff = max_mem_after - max_mem_before
            if not hasattr(self, '_edge_block_mem_logged'):
                print(f"EdgeBlock - Allocated: {mem_diff:.2f} MB, Peak increase: {max_mem_diff:.2f} MB")
                self._edge_block_mem_logged = True
        
        edge_attr = edge_attr + edge_attr_new
        
        # Process nodes
        node_attr_new = self.node_block(node_attr, edge_attr, edge_index)
        
        # Residual connections
        node_attr = node_attr + node_attr_new
        
        return node_attr, edge_attr