"""
Graph Distance-Biased Transolver implementation.

This extends Transolver++ by biasing the attention mechanism using precomputed graph
distances. Instead of modifying the slicing process, this directly penalizes attention
between topologically distant slices in the eidetic self-attention mechanism.

Key innovations:
- Precomputed k-hop shortest path distances between nodes
- Distance-biased attention over eidetic states (slices)
- Learnable distance penalty strength
- Efficient: graph distances are precomputed once during dataset loading

Note: Graph distances should be precomputed and stored in data.graph_distances
      during dataset loading. Use add_graph_distances() utility function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.utils import to_dense_batch
import numpy as np


ACTIVATION = {
    'gelu': nn.GELU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU,
    'leaky_relu': lambda: nn.LeakyReLU(0.1),
    'softplus': nn.Softplus,
    'elu': nn.ELU,
    'silu': nn.SiLU
}


def compute_graph_distances_khop(edge_index, num_nodes, max_hops=5):
    """
    Compute k-hop shortest path distances between all node pairs.

    Uses BFS to compute distances up to max_hops. Nodes beyond max_hops
    are assigned distance = max_hops + 1.

    This is a preprocessing function meant to be called once per graph during
    dataset loading, not during training.

    Args:
        edge_index: [2, num_edges] edge connectivity
        num_nodes: Number of nodes in the graph
        max_hops: Maximum hop distance to compute (default: 5)

    Returns:
        [num_nodes, num_nodes] distance matrix (values: 0 to max_hops+1)
    """
    # Build adjacency list for BFS
    adj_list = [[] for _ in range(num_nodes)]
    edge_index_np = edge_index.cpu().numpy()
    for i in range(edge_index_np.shape[1]):
        src, dst = edge_index_np[0, i], edge_index_np[1, i]
        adj_list[src].append(dst)

    # Initialize distance matrix with max_hops + 1 (unreachable within max_hops)
    distances = np.full((num_nodes, num_nodes), max_hops + 1, dtype=np.float32)
    np.fill_diagonal(distances, 0)  # Distance to self is 0

    # BFS from each node
    for start_node in range(num_nodes):
        queue = [(start_node, 0)]
        visited = {start_node}

        while queue:
            node, dist = queue.pop(0)

            if dist >= max_hops:
                continue

            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[start_node, neighbor] = dist + 1
                    queue.append((neighbor, dist + 1))

    return torch.from_numpy(distances)


def gumbel_softmax(logits, tau=1, hard=False):
    """
    Gumbel-Softmax sampling for differentiable discrete distributions.

    Args:
        logits: Unnormalized log probabilities
        tau: Temperature parameter (lower = more discrete)
        hard: If True, returns one-hot but backprops through soft sample
    """
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)

    y = logits + gumbel_noise
    y = y / tau
    y = F.softmax(y, dim=-1)

    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y

    return y


class DistanceBiasedPhysicsAttention(nn.Module):
    """
    Distance-biased physics attention mechanism with eidetic states.

    This module extends PhysicsAttentionEidetic by incorporating graph distance
    information to bias the self-attention over eidetic states. Attention between
    topologically distant slices is penalized.

    Complexity: O(N*G*C + G²*C) vs O(N²*C) for standard self-attention
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64,
                 distance_bias_type='exp', learnable_bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.slice_num = slice_num
        self.distance_bias_type = distance_bias_type

        # Temperature modulation for Gumbel-Softmax
        self.bias = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.proj_temperature = nn.Sequential(
            nn.Linear(dim_head, slice_num),
            nn.GELU(),
            nn.Linear(slice_num, 1),
            nn.GELU()
        )

        # Input projections
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)

        # Initialize slice projection with orthogonal weights
        torch.nn.init.orthogonal_(self.in_project_slice.weight)

        # QKV projections for eidetic states
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # Distance bias parameters
        if learnable_bias:
            if distance_bias_type == 'exp':
                # exp(-alpha * distance)
                self.distance_alpha = nn.Parameter(torch.tensor(0.5))
            elif distance_bias_type == 'linear':
                # -alpha * distance
                self.distance_alpha = nn.Parameter(torch.tensor(0.1))
            elif distance_bias_type == 'quadratic':
                # -alpha * distance^2
                self.distance_alpha = nn.Parameter(torch.tensor(0.05))
            else:
                raise ValueError(f"Unknown distance_bias_type: {distance_bias_type}")
        else:
            self.register_buffer('distance_alpha', torch.tensor(0.5))

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def compute_slice_distances(self, slice_weights, node_distances):
        """
        Compute pairwise distances between slices based on node assignments.

        Args:
            slice_weights: [B, H, N, G] soft assignment of nodes to slices
            node_distances: [B, N, N] pairwise node distances

        Returns:
            [B, H, G, G] pairwise slice distances
        """
        B, H, N, G = slice_weights.shape

        # For each slice pair (g1, g2), compute average distance between their member nodes
        # Distance(slice_i, slice_j) = sum_n sum_m (weight[n,i] * weight[m,j] * dist[n,m])
        #                             / (sum_n weight[n,i] * sum_m weight[m,j])

        # Expand dimensions for broadcasting
        slice_weights_i = slice_weights.unsqueeze(4)  # [B, H, N, G, 1]
        slice_weights_j = slice_weights.unsqueeze(3)  # [B, H, N, 1, G]

        # Expand node distances for heads
        node_distances_expanded = node_distances.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # [B, 1, N, 1, 1, N]

        # Compute weighted distances
        # For each head and slice pair, we need to compute weighted average of node distances
        slice_distances = torch.zeros(B, H, G, G, device=slice_weights.device)

        for b in range(B):
            for h in range(H):
                # Get slice assignments for this batch and head: [N, G]
                weights = slice_weights[b, h]  # [N, G]

                # Compute slice distance matrix
                # dist[i,j] = sum over all node pairs (n,m) of weights[n,i] * weights[m,j] * node_dist[n,m]
                #             normalized by sum of weights
                for i in range(G):
                    for j in range(G):
                        if i == j:
                            slice_distances[b, h, i, j] = 0.0
                        else:
                            # Weight for each node pair
                            weight_pairs = weights[:, i:i+1] @ weights[:, j:j+1].T  # [N, N]
                            weighted_dist = (weight_pairs * node_distances[b]).sum()
                            total_weight = weight_pairs.sum()
                            slice_distances[b, h, i, j] = weighted_dist / (total_weight + 1e-8)

        return slice_distances

    def compute_distance_bias(self, slice_distances):
        """
        Compute attention bias from slice distances.

        Args:
            slice_distances: [B, H, G, G] pairwise slice distances

        Returns:
            [B, H, G, G] attention bias (to be added to attention scores)
        """
        alpha = self.distance_alpha

        if self.distance_bias_type == 'exp':
            # Exponential decay: exp(-alpha * distance)
            # Convert to log space for numerical stability (will be added before softmax)
            bias = -alpha * slice_distances
        elif self.distance_bias_type == 'linear':
            # Linear penalty: -alpha * distance
            bias = -alpha * slice_distances
        elif self.distance_bias_type == 'quadratic':
            # Quadratic penalty: -alpha * distance^2
            bias = -alpha * (slice_distances ** 2)
        else:
            bias = torch.zeros_like(slice_distances)

        return bias

    def forward(self, x, node_distances=None, mask=None):
        """
        Args:
            x: [B, N, C] node features
            node_distances: [B, N, N] precomputed graph distances between nodes
            mask: [B, N] boolean mask (True for real nodes, False for padding)

        Returns:
            [B, N, C] refined node features
        """
        B, N, C = x.shape

        # Project to multi-head representation
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
        x_mid = x_mid.permute(0, 2, 1, 3).contiguous()  # [B, H, N, C_head]

        # Compute temperature-modulated assignment weights
        temperature = self.proj_temperature(x_mid) + self.bias
        temperature = torch.clamp(temperature, min=0.01)

        # Soft assignment of nodes to eidetic states
        slice_weights = gumbel_softmax(
            self.in_project_slice(x_mid),
            temperature
        )  # [B, H, N, G]

        # Apply mask if provided (for variable-size graphs)
        if mask is not None:
            mask_expanded = mask.view(B, 1, N, 1)  # [B, 1, N, 1]
            slice_weights = slice_weights * mask_expanded

        # Aggregate nodes into eidetic states
        slice_norm = slice_weights.sum(2)  # [B, H, G]
        slice_token = torch.einsum(
            "bhnc,bhng->bhgc",
            x_mid,
            slice_weights
        ).contiguous()  # [B, H, G, C_head]

        # Normalize by assignment mass
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        # Compute QKV for eidetic states
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)

        # Standard attention scores
        attn_scores = torch.einsum('bhqc,bhkc->bhqk', q_slice_token, k_slice_token) * self.scale

        # Add distance bias if available
        if node_distances is not None:
            slice_distances = self.compute_slice_distances(slice_weights, node_distances)
            distance_bias = self.compute_distance_bias(slice_distances)
            attn_scores = attn_scores + distance_bias

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out_slice_token = torch.einsum('bhqk,bhkc->bhqc', attn_weights, v_slice_token)

        # Broadcast back to original nodes
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')

        return self.to_out(out_x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with residual connections.
    """

    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act_fn = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation {act} not implemented")

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn())
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class DistanceBiasedTransolverBlock(nn.Module):
    """
    Transformer block with distance-biased attention and feed-forward network.

    Architecture:
        Input -> LayerNorm -> Distance-Biased Attention -> Residual
              -> LayerNorm -> MLP -> Residual -> Output
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act='gelu',
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        distance_bias_type='exp',
        learnable_bias=True,
        use_checkpoint=True,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_checkpoint = use_checkpoint

        # Pre-normalization
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        # Distance-biased attention
        self.attn = DistanceBiasedPhysicsAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            distance_bias_type=distance_bias_type,
            learnable_bias=learnable_bias
        )

        # Feed-forward network with expansion
        self.mlp = MLP(
            hidden_dim,
            hidden_dim * mlp_ratio,
            hidden_dim,
            n_layers=0,
            res=False,
            act=act
        )

        # Optional output projection for last layer
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx, node_distances=None, mask=None):
        """
        Args:
            fx: [B, N, C] node features
            node_distances: [B, N, N] graph distances between nodes
            mask: [B, N] boolean mask
        """
        # Attention block with residual
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            fx = checkpoint(
                self._attn_forward,
                fx,
                node_distances,
                mask,
                use_reentrant=False
            ) + fx
        else:
            fx = self._attn_forward(fx, node_distances, mask) + fx

        # MLP block with residual
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            fx = checkpoint(
                self._mlp_forward,
                fx,
                use_reentrant=False
            ) + fx
        else:
            fx = self._mlp_forward(fx) + fx

        # Optional output projection
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx

    def _attn_forward(self, fx, node_distances, mask):
        return self.attn(self.ln_1(fx), node_distances, mask)

    def _mlp_forward(self, fx):
        return self.mlp(self.ln_2(fx))


class GraphDistanceTransolver(nn.Module):
    """
    Graph Distance-Biased Transolver for aerodynamic mesh predictions.

    This model extends Transolver++ by biasing attention based on graph topology.
    Instead of modifying the slicing process, it directly penalizes attention between
    topologically distant slices, encouraging the model to focus on local neighborhoods.

    Key features:
    - Precomputed k-hop shortest path distances between nodes
    - Distance-biased self-attention over eidetic states
    - Learnable distance penalty strength
    - Multiple bias types: exponential, linear, quadratic
    - Compatible with PyTorch Geometric batching and variable-size graphs
    """

    def __init__(
        self,
        input_node_dim: int,
        output_node_dim: int,
        space_dim: int = 2,
        n_layers: int = 6,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = 'gelu',
        mlp_ratio: int = 4,
        slice_num: int = 32,
        distance_bias_type: str = 'exp',
        learnable_bias: bool = True,
        use_checkpoint: bool = True,
        fourier_features: bool = False,
        fourier_dim: int = 0,
    ):
        """
        Args:
            input_node_dim: Input feature dimension (includes positions, normals, params)
            output_node_dim: Output dimension (pressure, shear stress, temperature)
            space_dim: Spatial dimension (2 for airfoils, 3 for 3D geometries)
            n_layers: Number of transformer blocks
            n_hidden: Hidden dimension
            dropout: Dropout rate
            n_head: Number of attention heads
            act: Activation function
            mlp_ratio: MLP expansion ratio
            slice_num: Number of eidetic states
            distance_bias_type: Type of distance bias ('exp', 'linear', 'quadratic')
            learnable_bias: Whether distance bias strength is learnable
            use_checkpoint: Use gradient checkpointing to save memory
            fourier_features: Apply Fourier encoding to positions
            fourier_dim: Dimension of Fourier features (if enabled)
        """
        super().__init__()
        self.__name__ = 'GraphDistanceTransolver'

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.fourier_features = fourier_features
        self.fourier_dim = fourier_dim

        # Calculate input dimension for preprocessing MLP
        preprocess_input_dim = input_node_dim
        if fourier_features and fourier_dim > 0:
            preprocess_input_dim += space_dim * fourier_dim * 2  # sin + cos

        # Preprocessing MLP: projects input features to hidden dimension
        self.preprocess = MLP(
            preprocess_input_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act
        )

        # Learnable bias/placeholder
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        # Stack of distance-biased transformer blocks
        self.blocks = nn.ModuleList([
            DistanceBiasedTransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=output_node_dim,
                slice_num=slice_num,
                distance_bias_type=distance_bias_type,
                learnable_bias=learnable_bias,
                last_layer=(i == n_layers - 1),
                use_checkpoint=use_checkpoint,
            )
            for i in range(n_layers)
        ])

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights with truncated normal."""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fourier_encode(self, pos):
        """
        Apply Fourier feature encoding to positions.

        Args:
            pos: [B, N, space_dim] positions

        Returns:
            [B, N, space_dim * fourier_dim * 2] encoded positions
        """
        if not self.fourier_features or self.fourier_dim == 0:
            return None

        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(
            0, self.fourier_dim - 1, self.fourier_dim,
            device=pos.device
        ) * torch.pi

        # Compute sin/cos features
        pos_expanded = pos.unsqueeze(-1) * freq_bands  # [B, N, space_dim, fourier_dim]
        pos_sin = torch.sin(pos_expanded)
        pos_cos = torch.cos(pos_expanded)

        # Concatenate and flatten
        fourier_features = torch.cat([pos_sin, pos_cos], dim=-1)  # [B, N, space_dim, 2*fourier_dim]
        fourier_features = fourier_features.flatten(-2)  # [B, N, space_dim * 2 * fourier_dim]

        return fourier_features

    def forward(self, x, edge_attr=None, edge_index=None, batch=None, graph_distances=None):
        """
        Forward pass compatible with aero-gnn training pipeline.

        Args:
            x: [total_nodes, input_node_dim] node features (flattened batch)
            edge_attr: Not used (for compatibility with GNN models)
            edge_index: [2, num_edges] edge connectivity (for compatibility)
            batch: [total_nodes] batch assignment tensor
            graph_distances: Either:
                - List of [N_i, N_i] distance matrices (from GraphDistanceDataLoader)
                - Single [N, N] distance matrix (for single graph inference)

        Returns:
            [total_nodes, output_node_dim] predictions (flattened batch)
        """
        device = x.device

        # Handle single graph case
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Check if graph distances are provided
        if graph_distances is None:
            raise ValueError(
                "GraphDistanceTransolver requires precomputed graph distances. "
                "Please add graph distances to your dataset using add_graph_distances() "
                "from models.graphDistanceTransolver utility."
            )

        # Convert PyG flattened batch to dense batch format
        x_batched, mask = to_dense_batch(x, batch)  # [B, max_N, C], [B, max_N]
        B, N, C = x_batched.shape

        # Create batch-wise distance matrices
        batch_distances = torch.zeros(B, N, N, device=device)

        # Handle two cases: list of matrices (from custom DataLoader) or single matrix (inference)
        if isinstance(graph_distances, list):
            # From GraphDistanceDataLoader - list of [N_i, N_i] matrices
            for b, dist_matrix in enumerate(graph_distances):
                if dist_matrix is not None:
                    # Ensure on correct device
                    if dist_matrix.device != device:
                        dist_matrix = dist_matrix.to(device)
                    n_nodes = dist_matrix.size(0)
                    batch_distances[b, :n_nodes, :n_nodes] = dist_matrix
        else:
            # Single graph or old-style usage - single [N, N] matrix
            if graph_distances.device != device:
                graph_distances = graph_distances.to(device)

            # If it's a 2D matrix, treat as single graph
            if graph_distances.dim() == 2:
                n_nodes = graph_distances.size(0)
                batch_distances[0, :n_nodes, :n_nodes] = graph_distances
            else:
                raise ValueError(f"Unexpected graph_distances shape: {graph_distances.shape}")

        # Extract positions if needed for Fourier encoding
        if self.fourier_features and self.fourier_dim > 0:
            # Assume first space_dim features are positions
            pos = x_batched[:, :, :self.space_dim]
            fourier_feat = self.fourier_encode(pos)
            x_input = torch.cat([x_batched, fourier_feat], dim=-1)
        else:
            x_input = x_batched

        # Preprocess inputs
        fx = self.preprocess(x_input)
        fx = fx + self.placeholder[None, None, :]

        # Process through distance-biased transformer blocks
        for block in self.blocks:
            fx = block(fx, batch_distances, mask)

        # Convert back to flattened format for PyG
        fx_flat = fx[mask]  # [total_nodes, output_dim]

        return fx_flat


# ============================================================================
# Dataset preprocessing utilities
# ============================================================================

def add_graph_distances_to_data(data, max_hops=5):
    """
    Add precomputed graph distances to a PyG Data object.

    This should be called once per graph during dataset loading, not during training.

    Args:
        data: PyG Data object with edge_index attribute
        max_hops: Maximum hop distance to compute

    Returns:
        data: Modified Data object with graph_distances attribute added
    """
    if hasattr(data, 'graph_distances'):
        # Already has graph distances, skip
        return data

    num_nodes = data.x.size(0) if hasattr(data, 'x') else data.num_nodes

    # Compute graph distances on CPU (this is preprocessing, called once)
    edge_index_cpu = data.edge_index.cpu()
    graph_distances = compute_graph_distances_khop(
        edge_index_cpu,
        num_nodes,
        max_hops=max_hops
    )

    # Store on CPU to save GPU memory (will be moved to GPU during training)
    # Store as sparse tensor to save memory for large graphs
    data.graph_distances = graph_distances.cpu()

    return data


def add_graph_distances_to_dataset(dataset, max_hops=5, verbose=True):
    """
    Add precomputed graph distances to all graphs in a dataset.

    Args:
        dataset: PyG Dataset or list of Data objects
        max_hops: Maximum hop distance to compute
        verbose: Print progress

    Returns:
        dataset: Modified dataset with graph_distances added to all graphs
    """
    from tqdm import tqdm

    if verbose:
        print(f"\n=== Precomputing Graph Distances (max_hops={max_hops}) ===")
        iterator = tqdm(range(len(dataset)), desc="Computing graph distances")
    else:
        iterator = range(len(dataset))

    for i in iterator:
        data = dataset[i]
        add_graph_distances_to_data(data, max_hops)

    if verbose:
        print(f"✓ Graph distances added to {len(dataset)} graphs")

    return dataset


# ============================================================================
# Custom DataLoader for handling graph distance matrices
# ============================================================================

def graph_distance_collate(batch):
    """
    Custom collate function for batching graphs with distance matrices.

    The graph_distances attribute (NxN matrix) cannot be batched automatically
    by PyG's default collate, so we handle it separately.

    Args:
        batch: List of Data objects with graph_distances attribute

    Returns:
        Batched Data object with graph_distances stored separately
    """
    from torch_geometric.data import Batch

    # Extract graph_distances and remove from data objects temporarily
    graph_distances_list = []
    for data in batch:
        if hasattr(data, 'graph_distances'):
            graph_distances_list.append(data.graph_distances)
            # Temporarily remove to prevent batching error
            dist = data.graph_distances
            del data.graph_distances

    # Use default PyG batching for everything else
    batched_data = Batch.from_data_list(batch)

    # Re-add graph_distances to original data objects
    for i, data in enumerate(batch):
        if i < len(graph_distances_list):
            data.graph_distances = graph_distances_list[i]

    # Store distances in a way that can be accessed during forward pass
    # We'll reconstruct the batch-wise distance matrix in the model
    batched_data.graph_distances_list = graph_distances_list

    return batched_data


class GraphDistanceDataLoader(torch.utils.data.DataLoader):
    """
    Custom DataLoader for GraphDistanceTransolver that handles graph distance matrices.

    Usage:
        loader = GraphDistanceDataLoader(dataset, batch_size=16, shuffle=True)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=graph_distance_collate,
            **kwargs
        )
