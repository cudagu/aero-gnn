"""
Bistride operations for multi-scale graph neural networks.
Based on BSMS-GNN: https://github.com/Eydcao/BSMS-GNN
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add, scatter_mean
from collections import deque
import numpy as np


class BistridePooling:
    """
    Bistride pooling strategy that selects nodes at alternating BFS levels.

    Pools nodes on every other frontier of breadth-first search (BFS),
    avoiding manual coarsening and preserving graph connectivity.
    """

    @staticmethod
    def bfs_distance(edge_index, num_nodes, start_node):
        """
        Compute BFS distances from a starting node.

        Args:
            edge_index: [2, num_edges] edge connectivity
            num_nodes: Total number of nodes
            start_node: Starting node index

        Returns:
            distances: [num_nodes] BFS distance from start_node (-1 for unreachable)
        """
        distances = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        distances[start_node] = 0

        # Build adjacency list
        adj_list = [[] for _ in range(num_nodes)]
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            adj_list[src].append(dst)

        # BFS
        queue = deque([start_node])
        while queue:
            node = queue.popleft()
            current_dist = distances[node]

            for neighbor in adj_list[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)

        return distances

    @staticmethod
    def select_bistride_nodes(edge_index, num_nodes, pos=None):
        """
        Select nodes using bistride algorithm (alternating BFS levels).

        Args:
            edge_index: [2, num_edges] edge connectivity
            num_nodes: Total number of nodes
            pos: [num_nodes, pos_dim] node positions (optional, for seed selection)

        Returns:
            selected_indices: Indices of nodes to keep at coarser level
        """
        device = edge_index.device

        # Select seed node (center-most node if positions available)
        if pos is not None:
            center = pos.mean(dim=0)
            distances_to_center = torch.norm(pos - center, dim=1)
            seed_node = torch.argmin(distances_to_center).item()
        else:
            # Random seed or use node with highest degree
            degrees = torch.bincount(edge_index[0], minlength=num_nodes)
            seed_node = torch.argmax(degrees).item()

        # Compute BFS distances from seed
        bfs_dists = BistridePooling.bfs_distance(edge_index, num_nodes, seed_node)

        # Select nodes at even distances (bistride)
        selected_mask = (bfs_dists % 2 == 0) & (bfs_dists >= 0)
        selected_indices = torch.where(selected_mask)[0]

        # If selection is too aggressive, also include odd levels
        if len(selected_indices) < num_nodes * 0.3:
            selected_mask = bfs_dists >= 0
            selected_indices = torch.where(selected_mask)[0]

        return selected_indices


class Unpool(nn.Module):
    """Unpooling operation to restore node features to original resolution."""

    def __init__(self):
        super().__init__()

    def forward(self, x_coarse, indices, num_nodes_fine):
        """
        Unpool features from coarse to fine level.

        Args:
            x_coarse: [num_coarse_nodes, dim] or [batch, num_coarse_nodes, dim]
            indices: Node indices that were kept during pooling
            num_nodes_fine: Number of nodes at fine level

        Returns:
            x_fine: [num_nodes_fine, dim] or [batch, num_nodes_fine, dim]
        """
        if x_coarse.dim() == 2:
            # Unbatched: [num_nodes, dim]
            device = x_coarse.device
            x_fine = torch.zeros(num_nodes_fine, x_coarse.shape[1],
                                device=device, dtype=x_coarse.dtype)
            x_fine[indices] = x_coarse
        else:
            # Batched: [batch, num_nodes, dim]
            device = x_coarse.device
            batch_size = x_coarse.shape[0]
            x_fine = torch.zeros(batch_size, num_nodes_fine, x_coarse.shape[2],
                                device=device, dtype=x_coarse.dtype)
            x_fine[:, indices, :] = x_coarse

        return x_fine


class WeightedEdgeConv(nn.Module):
    """
    Edge convolution with learned edge weights for message passing.
    """

    def __init__(self, in_dim, out_dim, aggr='add'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.aggr = aggr

        # MLP for computing edge weights
        self.edge_weight_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, 64),  # src + dst + edge_length
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.transform = nn.Linear(in_dim, out_dim)

    def compute_edge_weights(self, x, edge_index, pos):
        """
        Compute edge weights based on node features and positions.

        Args:
            x: [num_nodes, in_dim] node features
            edge_index: [2, num_edges]
            pos: [num_nodes, pos_dim] node positions

        Returns:
            edge_weights: [num_edges, 1]
        """
        src, dst = edge_index

        # Edge features: concatenate src, dst features and edge length
        edge_length = torch.norm(pos[dst] - pos[src], dim=1, keepdim=True)
        edge_feat = torch.cat([x[src], x[dst], edge_length], dim=1)

        weights = self.edge_weight_mlp(edge_feat)
        return weights

    def forward(self, x, edge_index, pos, edge_weights=None, compute_weights=True):
        """
        Forward pass with weighted edge convolution.

        Args:
            x: [num_nodes, in_dim] node features
            edge_index: [2, num_edges]
            pos: [num_nodes, pos_dim] positions
            edge_weights: Pre-computed edge weights (optional)
            compute_weights: Whether to compute new weights

        Returns:
            out: [num_nodes, out_dim] updated node features
            edge_weights: [num_edges, 1] edge weights (if computed)
        """
        src, dst = edge_index

        # Compute or use provided edge weights
        if compute_weights and edge_weights is None:
            edge_weights = self.compute_edge_weights(x, edge_index, pos)

        # Transform features
        x_transformed = self.transform(x)

        # Weighted message passing
        messages = x_transformed[src] * edge_weights

        # Aggregate messages
        if self.aggr == 'add':
            out = scatter_add(messages, dst, dim=0, dim_size=x.shape[0])
        elif self.aggr == 'mean':
            out = scatter_mean(messages, dst, dim=0, dim_size=x.shape[0])
        else:
            raise ValueError(f"Unknown aggregation: {self.aggr}")

        return out, edge_weights


class GMP(nn.Module):
    """
    Graph Message Passing layer with edge and node updates.
    """

    def __init__(self, node_dim, edge_dim, hidden_dim, activation='relu'):
        super().__init__()

        # Edge update MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
            nn.Linear(hidden_dim, edge_dim),
            nn.LayerNorm(edge_dim)
        )

        # Node update MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
            nn.LayerNorm(node_dim)
        )

    def forward(self, x, edge_attr, edge_index):
        """
        Args:
            x: [num_nodes, node_dim]
            edge_attr: [num_edges, edge_dim]
            edge_index: [2, num_edges]

        Returns:
            x_updated: [num_nodes, node_dim]
            edge_attr_updated: [num_edges, edge_dim]
        """
        src, dst = edge_index

        # Update edges
        edge_input = torch.cat([x[src], x[dst], edge_attr], dim=1)
        edge_attr_new = self.edge_mlp(edge_input)
        edge_attr = edge_attr + edge_attr_new  # Residual

        # Aggregate edges to nodes
        edge_aggr = scatter_add(edge_attr, dst, dim=0, dim_size=x.shape[0])

        # Update nodes
        node_input = torch.cat([x, edge_aggr], dim=1)
        x_new = self.node_mlp(node_input)
        x = x + x_new  # Residual

        return x, edge_attr
