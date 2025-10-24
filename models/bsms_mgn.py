import torch
from torch import nn
from torch_scatter import scatter_mean

from models.mlp import MLP
from models.mgnLayer import MeshGraphNetLayer


class BiStridedMeshGraphNet(nn.Module):
    """Multi-scale MeshGraphNet using bi-strided pooling and skip connections."""

    def __init__(
        self,
        input_node_dim: int,
        input_edge_dim: int,
        output_node_dim: int,
        processor_size: int = 15,
        activation_fn: str = "relu",
        num_hidden_layers_node_processor: int = 1,
        num_hidden_layers_edge_processor: int = 1,
        hidden_dim_processor: int = 128,
        num_hidden_layers_node_encoder: int = 1,
        hidden_dim_node_encoder: int = 128,
        num_hidden_layers_edge_encoder: int = 1,
        hidden_dim_edge_encoder: int = 128,
        aggregation: str = "add",
        hidden_dim_decoder: int = 128,
        num_hidden_layers_decoder: int = 1,
        dropout: float = 0.0,
        do_concat_trick: bool = False,
        num_scales: int = 3,
        layers_per_scale: int = 2,
        stride: int = 2,
    ) -> None:
        super().__init__()

        if num_scales < 1:
            raise ValueError("num_scales must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")

        self.num_scales = num_scales
        self.stride = stride
        self.aggregation = aggregation
        self.do_concat_trick = do_concat_trick

        # Encoder projections
        self.node_encoder = MLP(
            input_dim=input_node_dim,
            hidden_dim=hidden_dim_node_encoder,
            output_dim=hidden_dim_processor,
            num_hidden_layers=num_hidden_layers_node_encoder,
            activation_fn=activation_fn,
            dropout=dropout,
            use_layer_norm=True,
        )

        self.edge_encoder = MLP(
            input_dim=input_edge_dim,
            hidden_dim=hidden_dim_edge_encoder,
            output_dim=hidden_dim_processor,
            num_hidden_layers=num_hidden_layers_edge_encoder,
            activation_fn=activation_fn,
            dropout=dropout,
            use_layer_norm=True,
        )

        # Determine how many layers are used per stage
        if isinstance(layers_per_scale, int):
            down_counts = [layers_per_scale for _ in range(max(num_scales - 1, 0))]
            up_counts = [layers_per_scale for _ in range(max(num_scales - 1, 0))]
        else:
            if len(layers_per_scale) != max(num_scales - 1, 0):
                raise ValueError(
                    "layers_per_scale must be int or list with num_scales-1 elements"
                )
            down_counts = list(layers_per_scale)
            up_counts = list(layers_per_scale)

        used_layers = 2 * sum(down_counts)
        bottleneck_layers = max(1, processor_size - used_layers)

        def _build_block(num_layers: int) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    MeshGraphNetLayer(
                        node_dim=hidden_dim_processor,
                        edge_dim=hidden_dim_processor,
                        hidden_dim=hidden_dim_processor,
                        num_hidden_layers_node_processor=num_hidden_layers_node_processor,
                        num_hidden_layers_edge_processor=num_hidden_layers_edge_processor,
                        activation_fn=activation_fn,
                        use_layer_norm=True,
                        aggregation=aggregation,
                        do_concat_trick=do_concat_trick,
                    )
                    for _ in range(num_layers)
                ]
            )

        # Down-sampling stacks
        self.down_layers = nn.ModuleList([
            _build_block(count) for count in down_counts
        ])

        # Bottleneck layers (operate on the coarsest graph)
        self.bottleneck_layers = _build_block(bottleneck_layers)

        # Up-sampling stacks (mirror of down path)
        self.up_layers = nn.ModuleList([
            _build_block(count) for count in reversed(up_counts)
        ])

        # Decoder
        self.decoder = MLP(
            input_dim=hidden_dim_processor,
            hidden_dim=hidden_dim_decoder,
            output_dim=output_node_dim,
            num_hidden_layers=num_hidden_layers_decoder,
            activation_fn=activation_fn,
            use_layer_norm=False,
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor = None,
        pos: torch.Tensor = None,
    ) -> torch.Tensor:
        if batch is None:
            batch = node_attr.new_zeros(node_attr.size(0), dtype=torch.long)

        # Encode features
        node_hidden = self.node_encoder(node_attr)
        edge_hidden = self.edge_encoder(edge_attr)

        if self.dropout is not None:
            node_hidden = self.dropout(node_hidden)
            edge_hidden = self.dropout(edge_hidden)

        assignments = []
        skip_connections = []

        current_batch = batch
        current_pos = pos
        current_edge_index = edge_index
        current_edge_hidden = edge_hidden
        current_node_hidden = node_hidden

        # Down-sampling path
        for scale_idx, layers in enumerate(self.down_layers):
            for layer in layers:
                current_node_hidden, current_edge_hidden = layer(
                    current_node_hidden, current_edge_hidden, current_edge_index
                )

            skip_connections.append(
                (
                    current_node_hidden,
                    current_edge_hidden,
                    current_edge_index,
                    current_batch,
                    current_pos,
                )
            )

            (
                current_node_hidden,
                current_edge_hidden,
                current_edge_index,
                current_batch,
                current_pos,
                assignment,
            ) = self._downsample(
                current_node_hidden,
                current_edge_hidden,
                current_edge_index,
                current_batch,
                current_pos,
            )
            assignments.append(assignment)

        # Bottleneck processing on the coarsest scale
        for layer in self.bottleneck_layers:
            current_node_hidden, current_edge_hidden = layer(
                current_node_hidden, current_edge_hidden, current_edge_index
            )

        # Up-sampling path
        for scale_idx, layers in enumerate(self.up_layers):
            assignment = assignments[-(scale_idx + 1)] if assignments else None
            if assignment is not None:
                skip_node, skip_edge, skip_edge_index, skip_batch, skip_pos = skip_connections[-(scale_idx + 1)]
                # Unpool coarse node features back to the finer resolution
                current_node_hidden = self._unpool_nodes(current_node_hidden, assignment)
                current_node_hidden = current_node_hidden + skip_node

                # Restore fine-scale connectivity information
                current_edge_hidden = skip_edge
                current_edge_index = skip_edge_index
                current_batch = skip_batch
                current_pos = skip_pos

            for layer in layers:
                current_node_hidden, current_edge_hidden = layer(
                    current_node_hidden, current_edge_hidden, current_edge_index
                )

        # Decode predictions
        predictions = self.decoder(current_node_hidden)
        return predictions

    def _downsample(
        self,
        node_attr: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        pos: torch.Tensor = None,
    ):
        device = node_attr.device
        num_nodes = node_attr.size(0)

        fine_to_coarse = torch.empty(num_nodes, dtype=torch.long, device=device)
        coarse_batch_chunks = []

        unique_batches = torch.unique_consecutive(batch)
        coarse_offset = 0

        for graph_id in unique_batches.tolist():
            mask = batch == graph_id
            indices = torch.nonzero(mask, as_tuple=False).view(-1)
            if indices.numel() == 0:
                continue

            if pos is not None:
                graph_pos = pos[indices]
                sort_idx = torch.argsort(graph_pos[:, 0])
                sorted_indices = indices[sort_idx]
            else:
                sorted_indices = indices

            local_count = sorted_indices.numel()
            local_indices = torch.arange(local_count, device=device)
            coarse_local = local_indices // self.stride
            num_coarse_nodes = int(coarse_local[-1].item() + 1)

            fine_to_coarse[sorted_indices] = coarse_local + coarse_offset
            coarse_batch_chunks.append(
                torch.full((num_coarse_nodes,), graph_id, device=device, dtype=torch.long)
            )
            coarse_offset += num_coarse_nodes

        coarse_batch = (
            torch.cat(coarse_batch_chunks, dim=0)
            if coarse_batch_chunks
            else torch.empty((0,), dtype=torch.long, device=device)
        )
        total_coarse_nodes = coarse_batch.size(0)

        coarse_node_attr = scatter_mean(
            node_attr, fine_to_coarse, dim=0, dim_size=total_coarse_nodes
        )

        if pos is not None:
            coarse_pos = scatter_mean(
                pos, fine_to_coarse, dim=0, dim_size=total_coarse_nodes
            )
        else:
            coarse_pos = None

        row, col = edge_index
        coarse_row = fine_to_coarse[row]
        coarse_col = fine_to_coarse[col]
        edge_keys = coarse_row * max(total_coarse_nodes, 1) + coarse_col
        unique_keys, inverse = torch.unique(edge_keys, return_inverse=True)

        if unique_keys.numel() > 0:
            coarse_edge_attr = scatter_mean(edge_attr, inverse, dim=0)
            coarse_edge_row = unique_keys // max(total_coarse_nodes, 1)
            coarse_edge_col = unique_keys % max(total_coarse_nodes, 1)
            coarse_edge_index = torch.stack(
                [coarse_edge_row, coarse_edge_col], dim=0
            )
        else:
            feature_dim = edge_attr.size(1) if edge_attr.dim() > 1 else 1
            coarse_edge_attr = edge_attr.new_zeros((0, feature_dim))
            coarse_edge_index = edge_index.new_zeros((2, 0))

        return (
            coarse_node_attr,
            coarse_edge_attr,
            coarse_edge_index,
            coarse_batch,
            coarse_pos,
            fine_to_coarse,
        )

    def _unpool_nodes(
        self, coarse_nodes: torch.Tensor, assignment: torch.Tensor
    ) -> torch.Tensor:
        return coarse_nodes[assignment]
