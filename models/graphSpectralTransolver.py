"""
Graph-Aware Spectral Transolver implementation.

This extends Transolver++ with graph structure awareness by using graph Laplacian
eigenvectors to inform node-to-slice assignments, resulting in more structurally
coherent clusters compared to purely feature-based Gumbel-softmax assignment.

Key innovations:
- Spectral clustering using normalized graph Laplacian eigenvectors (precomputed)
- Graph-aware slice assignment combining spectral and feature information
- Maintains O(N*G*C + G²*C) complexity while respecting mesh topology
- Efficient: spectral features are precomputed once during dataset loading

Note: Spectral features should be precomputed and stored in data.spectral_features
      during dataset loading. Use add_spectral_features() utility function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_geometric.utils import to_dense_batch


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


def compute_graph_laplacian_eigenvectors_torch(edge_index, num_nodes, k=8, normalization='sym'):
    """
    Compute the first k eigenvectors of the graph Laplacian using PyTorch (GPU-friendly).

    This is a preprocessing function meant to be called once per graph during dataset
    loading, not during training.

    Args:
        edge_index: [2, num_edges] edge connectivity
        num_nodes: Number of nodes in the graph
        k: Number of eigenvectors to compute
        normalization: 'sym' for symmetric normalization, 'rw' for random walk

    Returns:
        [num_nodes, k] eigenvector matrix
    """
    from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
    import scipy.sparse.linalg as splinalg

    device = edge_index.device

    # Get normalized Laplacian in PyG format
    edge_index_lap, edge_weight_lap = get_laplacian(
        edge_index,
        normalization=normalization,
        num_nodes=num_nodes
    )

    # For preprocessing (called once), using scipy on CPU is acceptable
    # Convert to scipy sparse matrix for stable eigendecomposition
    try:
        L_sparse = to_scipy_sparse_matrix(edge_index_lap, edge_weight_lap, num_nodes)

        # Compute smallest k eigenvectors (corresponding to smallest eigenvalues)
        # These capture the graph's low-frequency structure
        k_actual = min(k, num_nodes - 2)
        eigenvalues, eigenvectors = splinalg.eigsh(
            L_sparse,
            k=k_actual,
            which='SM',  # Smallest magnitude
            return_eigenvectors=True
        )

        # Pad with zeros if we got fewer eigenvectors than requested
        if k_actual < k:
            padding = torch.zeros(num_nodes, k - k_actual)
            eigenvectors = torch.cat([
                torch.from_numpy(eigenvectors.astype('float32')),
                padding
            ], dim=1)
        else:
            eigenvectors = torch.from_numpy(eigenvectors.astype('float32'))

    except Exception as e:
        # Fallback to random features if eigendecomposition fails
        print(f"Warning: Eigendecomposition failed for graph with {num_nodes} nodes ({e}), using random features")
        eigenvectors = torch.randn(num_nodes, k)
        eigenvectors = F.normalize(eigenvectors, dim=0)

    return eigenvectors.to(device)


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


class SpectralPhysicsAttention(nn.Module):
    """
    Spectral graph-aware attention mechanism with eidetic states.

    This module extends PhysicsAttentionEidetic by incorporating graph Laplacian
    eigenvectors into the slice assignment process, creating structurally coherent
    clusters that respect the mesh topology.

    Complexity: O(N*G*C + G²*C) vs O(N²*C) for standard self-attention
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64,
                 spectral_dim=8, spectral_weight=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.spectral_dim = spectral_dim
        self.spectral_weight = spectral_weight

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

        # Feature-based slice projection
        self.in_project_slice_features = nn.Linear(dim_head, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice_features.weight)

        # Spectral-based slice projection
        self.in_project_slice_spectral = nn.Linear(spectral_dim, slice_num)
        torch.nn.init.orthogonal_(self.in_project_slice_spectral.weight)

        # Learnable weight for combining feature and spectral information
        self.spectral_gate = nn.Parameter(torch.tensor(spectral_weight))

        # QKV projections for eidetic states
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, spectral_features, mask=None):
        """
        Args:
            x: [B, N, C] node features
            spectral_features: [B, N, spectral_dim] graph Laplacian eigenvectors
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

        # Feature-based slice logits
        slice_logits_features = self.in_project_slice_features(x_mid)  # [B, H, N, G]

        # Spectral-based slice logits (broadcast spectral features across heads)
        spectral_expanded = spectral_features.unsqueeze(1).expand(-1, self.heads, -1, -1)  # [B, H, N, spectral_dim]
        slice_logits_spectral = self.in_project_slice_spectral(spectral_expanded)  # [B, H, N, G]

        # Combine feature and spectral information with learnable gating
        gate = torch.sigmoid(self.spectral_gate)
        slice_logits = (1 - gate) * slice_logits_features + gate * slice_logits_spectral

        # Soft assignment of nodes to eidetic states with Gumbel-Softmax
        slice_weights = gumbel_softmax(slice_logits, temperature)  # [B, H, N, G]

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

        # Self-attention over eidetic states
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        out_slice_token = F.scaled_dot_product_attention(
            q_slice_token,
            k_slice_token,
            v_slice_token
        )

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


class SpectralTransolverBlock(nn.Module):
    """
    Transformer block with spectral graph-aware attention and feed-forward network.

    Architecture:
        Input -> LayerNorm -> Spectral Attention -> Residual
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
        spectral_dim=8,
        spectral_weight=0.5,
        use_checkpoint=True,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_checkpoint = use_checkpoint

        # Pre-normalization
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        # Spectral graph-aware attention
        self.attn = SpectralPhysicsAttention(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            spectral_dim=spectral_dim,
            spectral_weight=spectral_weight
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

    def forward(self, fx, spectral_features, mask=None):
        """
        Args:
            fx: [B, N, C] node features
            spectral_features: [B, N, spectral_dim] graph Laplacian eigenvectors
            mask: [B, N] boolean mask
        """
        # Attention block with residual
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            fx = checkpoint(
                self._attn_forward,
                fx,
                spectral_features,
                mask,
                use_reentrant=False
            ) + fx
        else:
            fx = self._attn_forward(fx, spectral_features, mask) + fx

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

    def _attn_forward(self, fx, spectral_features, mask):
        return self.attn(self.ln_1(fx), spectral_features, mask)

    def _mlp_forward(self, fx):
        return self.mlp(self.ln_2(fx))


class GraphSpectralTransolver(nn.Module):
    """
    Graph-Aware Spectral Transolver for aerodynamic mesh predictions.

    This model extends Transolver++ with graph structure awareness by using
    graph Laplacian eigenvectors to inform node-to-slice assignments. This
    creates more structurally coherent clusters that respect mesh topology,
    potentially improving predictions on irregular meshes.

    Key features:
    - Spectral clustering using normalized graph Laplacian eigenvectors
    - Graph-aware slice assignment combining spectral and feature information
    - Learnable gating between spectral and feature-based clustering
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
        spectral_dim: int = 8,
        spectral_weight: float = 0.5,
        use_checkpoint: bool = True,
        fourier_features: bool = False,
        fourier_dim: int = 0,
        laplacian_norm: str = 'sym',
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
            spectral_dim: Number of Laplacian eigenvectors to use
            spectral_weight: Initial weight for spectral features (0=feature-only, 1=spectral-only)
            use_checkpoint: Use gradient checkpointing to save memory
            fourier_features: Apply Fourier encoding to positions
            fourier_dim: Dimension of Fourier features (if enabled)
            laplacian_norm: Laplacian normalization ('sym' or 'rw')
        """
        super().__init__()
        self.__name__ = 'GraphSpectralTransolver'

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.fourier_features = fourier_features
        self.fourier_dim = fourier_dim
        self.spectral_dim = spectral_dim
        self.laplacian_norm = laplacian_norm

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

        # Spectral feature preprocessing
        self.spectral_preprocess = nn.Linear(spectral_dim, spectral_dim)

        # Learnable bias/placeholder
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        # Stack of spectral transformer blocks
        self.blocks = nn.ModuleList([
            SpectralTransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=output_node_dim,
                slice_num=slice_num,
                spectral_dim=spectral_dim,
                spectral_weight=spectral_weight,
                last_layer=(i == n_layers - 1),
                use_checkpoint=use_checkpoint,
            )
            for i in range(n_layers)
        ])

        # Cache for spectral features (computed once per graph)
        self.register_buffer('_cached_spectral_features', None)
        self.register_buffer('_cached_batch_ptr', None)

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

    def forward(self, x, edge_attr=None, edge_index=None, batch=None, spectral_features=None):
        """
        Forward pass compatible with aero-gnn training pipeline.

        Args:
            x: [total_nodes, input_node_dim] node features (flattened batch)
            edge_attr: Not used (for compatibility with GNN models)
            edge_index: [2, num_edges] edge connectivity (for compatibility)
            batch: [total_nodes] batch assignment tensor
            spectral_features: [total_nodes, spectral_dim] precomputed spectral features
                              If None, will look for it in the data object passed to DataLoader

        Returns:
            [total_nodes, output_node_dim] predictions (flattened batch)
        """
        device = x.device

        # Handle single graph case
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # Check if spectral features are provided
        if spectral_features is None:
            raise ValueError(
                "GraphSpectralTransolver requires precomputed spectral features. "
                "Please add spectral features to your dataset using add_spectral_features() "
                "from models.graphSpectralTransolver utility."
            )

        # Ensure spectral features are on the right device
        if spectral_features.device != device:
            spectral_features = spectral_features.to(device)

        # Preprocess spectral features
        spectral_features_flat = self.spectral_preprocess(spectral_features)

        # Convert PyG flattened batch to dense batch format
        x_batched, mask = to_dense_batch(x, batch)  # [B, max_N, C], [B, max_N]
        spectral_batched, _ = to_dense_batch(spectral_features_flat, batch)  # [B, max_N, spectral_dim]

        B, N, C = x_batched.shape

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

        # Process through spectral transformer blocks
        for block in self.blocks:
            fx = block(fx, spectral_batched, mask)

        # Convert back to flattened format for PyG
        fx_flat = fx[mask]  # [total_nodes, output_dim]

        return fx_flat


def create_graph_spectral_transolver_model(config, input_node_dim, output_node_dim, space_dim):
    """
    Factory function to create GraphSpectralTransolver model from config dict.

    Args:
        config: Model configuration dictionary
        input_node_dim: Input feature dimension
        output_node_dim: Output dimension
        space_dim: Spatial dimension (2 or 3)

    Returns:
        GraphSpectralTransolver model instance
    """
    return GraphSpectralTransolver(
        input_node_dim=input_node_dim,
        output_node_dim=output_node_dim,
        space_dim=space_dim,
        n_layers=config.get('n_layers', 6),
        n_hidden=config.get('n_hidden', 256),
        dropout=config.get('dropout', 0.0),
        n_head=config.get('n_head', 8),
        act=config.get('act', 'gelu'),
        mlp_ratio=config.get('mlp_ratio', 4),
        slice_num=config.get('slice_num', 32),
        spectral_dim=config.get('spectral_dim', 8),
        spectral_weight=config.get('spectral_weight', 0.5),
        use_checkpoint=config.get('use_checkpoint', True),
        fourier_features=config.get('fourier_features', False),
        fourier_dim=config.get('fourier_dim', 0),
        laplacian_norm=config.get('laplacian_norm', 'sym'),
    )


# ============================================================================
# Dataset preprocessing utilities
# ============================================================================

def add_spectral_features_to_data(data, spectral_dim=8, laplacian_norm='sym'):
    """
    Add precomputed spectral features to a PyG Data object.

    This should be called once per graph during dataset loading, not during training.

    Args:
        data: PyG Data object with edge_index attribute
        spectral_dim: Number of Laplacian eigenvectors to compute
        laplacian_norm: Normalization type ('sym' or 'rw')

    Returns:
        data: Modified Data object with spectral_features attribute added
    """
    if hasattr(data, 'spectral_features'):
        # Already has spectral features, skip
        return data

    num_nodes = data.x.size(0) if hasattr(data, 'x') else data.num_nodes

    # Compute spectral features on CPU (this is preprocessing, called once)
    edge_index_cpu = data.edge_index.cpu()
    spectral_features = compute_graph_laplacian_eigenvectors_torch(
        edge_index_cpu,
        num_nodes,
        k=spectral_dim,
        normalization=laplacian_norm
    )

    # Store on CPU to save GPU memory (will be moved to GPU during training)
    data.spectral_features = spectral_features.cpu()

    return data


def add_spectral_features_to_dataset(dataset, spectral_dim=8, laplacian_norm='sym', verbose=True):
    """
    Add precomputed spectral features to all graphs in a dataset.

    Args:
        dataset: PyG Dataset or list of Data objects
        spectral_dim: Number of Laplacian eigenvectors to compute
        laplacian_norm: Normalization type ('sym' or 'rw')
        verbose: Print progress

    Returns:
        dataset: Modified dataset with spectral_features added to all graphs
    """
    from tqdm import tqdm

    if verbose:
        print(f"\n=== Precomputing Spectral Features (k={spectral_dim}) ===")
        iterator = tqdm(range(len(dataset)), desc="Computing spectral features")
    else:
        iterator = range(len(dataset))

    for i in iterator:
        data = dataset[i]
        add_spectral_features_to_data(data, spectral_dim, laplacian_norm)

    if verbose:
        print(f"✓ Spectral features added to {len(dataset)} graphs")

    return dataset
