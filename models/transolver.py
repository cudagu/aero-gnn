"""
Transolver++ implementation adapted for aero-gnn framework.

Based on "Transolver++: Scaling Neural Operators to Massive Geometries" (ICML 2025)
Original repository: https://github.com/thuml/Transolver_plus

Key adaptations:
- Removed distributed training dependencies for single-GPU use
- Integrated with PyTorch Geometric batching (variable-size graphs)
- Added support for 2D/3D aerodynamic meshes
- Compatible with aero-gnn training pipeline
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


class PhysicsAttentionEidetic(nn.Module):
    """
    Physics-aware attention mechanism with eidetic states.

    This module compresses N mesh points into G eidetic states (G << N),
    performs self-attention on the compressed representation, then broadcasts
    back to the original mesh points.

    Complexity: O(N*G*C + G²*C) vs O(N²*C) for standard self-attention
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dropout = nn.Dropout(dropout)

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

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, C] node features
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


class TransolverBlock(nn.Module):
    """
    Transformer block with eidetic attention and feed-forward network.

    Architecture:
        Input -> LayerNorm -> Eidetic Attention -> Residual
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
        use_checkpoint=True,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.use_checkpoint = use_checkpoint

        # Pre-normalization
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.ln_2 = nn.LayerNorm(hidden_dim)

        # Eidetic attention
        self.attn = PhysicsAttentionEidetic(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num
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

    def forward(self, fx, mask=None):
        """
        Args:
            fx: [B, N, C] node features
            mask: [B, N] boolean mask
        """
        # Attention block with residual
        if self.training and self.use_checkpoint:
            from torch.utils.checkpoint import checkpoint
            fx = checkpoint(
                self._attn_forward,
                fx,
                mask,
                use_reentrant=False
            ) + fx
        else:
            fx = self._attn_forward(fx, mask) + fx

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

    def _attn_forward(self, fx, mask):
        return self.attn(self.ln_1(fx), mask)

    def _mlp_forward(self, fx):
        return self.mlp(self.ln_2(fx))


class TransolverAero(nn.Module):
    """
    Transolver++ model adapted for aerodynamic mesh predictions.

    This model processes unstructured meshes using transformer attention
    with eidetic state compression, suitable for large-scale aerodynamic
    simulations on 2D airfoils and 3D geometries.

    Key differences from original Transolver:
    - Removed hardcoded unified position encoding
    - Added support for PyTorch Geometric batching
    - Integrated with aero-gnn's feature handling
    - Optional Fourier position encoding
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
        use_checkpoint: bool = True,
        fourier_features: bool = False,
        fourier_dim: int = 0,
        condition_dim: int = 0,
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
            use_checkpoint: Use gradient checkpointing to save memory
            fourier_features: Apply Fourier encoding to positions
            fourier_dim: Dimension of Fourier features (if enabled)
            condition_dim: Dimension of global condition vector (e.g., [mach, alpha])
        """
        super().__init__()
        self.__name__ = 'TransolverAero'

        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.fourier_features = fourier_features
        self.fourier_dim = fourier_dim
        self.condition_dim = condition_dim

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

        # Optional condition embedding for global parameters
        if condition_dim > 0:
            self.embedding = nn.Linear(condition_dim, n_hidden)

        # Learnable bias/placeholder
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransolverBlock(
                num_heads=n_head,
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=output_node_dim,
                slice_num=slice_num,
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

    def forward(self, x, edge_attr=None, edge_index=None, batch=None):
        """
        Forward pass compatible with aero-gnn training pipeline.

        Args:
            x: [total_nodes, input_node_dim] node features (flattened batch)
            edge_attr: Not used (for compatibility with GNN models)
            edge_index: Not used (for compatibility with GNN models)
            batch: [total_nodes] batch assignment tensor

        Returns:
            [total_nodes, output_node_dim] predictions (flattened batch)
        """
        # Convert PyG flattened batch to dense batch format
        if batch is not None:
            x_batched, mask = to_dense_batch(x, batch)  # [B, max_N, C], [B, max_N]
        else:
            # Single graph case
            x_batched = x.unsqueeze(0)  # [1, N, C]
            mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)

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

        # Add condition embedding if available
        # Note: In aero-gnn, global params are broadcast to all nodes
        # So we could extract them from x_batched if needed
        # For now, we skip this since params are already in x

        # Process through transformer blocks
        for block in self.blocks:
            fx = block(fx, mask)

        # Convert back to flattened format for PyG
        if batch is not None:
            # Extract only real nodes (not padding)
            fx_flat = fx[mask]  # [total_nodes, output_dim]
        else:
            fx_flat = fx.squeeze(0)  # [N, output_dim]

        return fx_flat


def create_transolver_model(config, input_node_dim, output_node_dim, space_dim):
    """
    Factory function to create Transolver model from config dict.

    Args:
        config: Model configuration dictionary
        input_node_dim: Input feature dimension
        output_node_dim: Output dimension
        space_dim: Spatial dimension (2 or 3)

    Returns:
        TransolverAero model instance
    """
    return TransolverAero(
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
        use_checkpoint=config.get('use_checkpoint', True),
        fourier_features=config.get('fourier_features', False),
        fourier_dim=config.get('fourier_dim', 0),
        condition_dim=config.get('condition_dim', 0),
    )
