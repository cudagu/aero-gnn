import torch
import torch.optim as optim
import numpy as np

def train(model, loader, optimizer, loss_fn, device, use_amp=False, amp_dtype=None, profiler=None):
    """Train the model for one epoch.

    Args:
        model: The model to train
        loader: DataLoader for training data
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (e.g., torch.bfloat16)
        profiler: Optional PyTorch profiler instance
    """
    model.train()
    total_loss = 0.0

    # Determine device type for autocast
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for batch in loader:
        batch = batch.to(device)

        # Check if model needs batch parameter (for poolMGN and MeshGraphNet_v2)
        model_class = model.__class__.__name__

        # Use autocast if AMP is enabled
        if use_amp and amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                if model_class == 'BSMS_MeshGraphNet':
                    # BSMS model needs multi_data dict
                    multi_data = {}
                    for key, value in batch.multi_data.items():
                        if isinstance(value, list):
                            multi_data[key] = [v.to(device) if torch.is_tensor(v) else v for v in value]
                        else:
                            multi_data[key] = value.to(device) if torch.is_tensor(value) else value
                    pred = model(batch.x, batch.edge_attr, batch.edge_index, multi_data)

                elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                    pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

                elif model_class in ['MLPNet']:
                    pred = model(batch.x)

                elif model_class == 'TransolverAero':
                    # Transolver uses batch tensor for PyG batching
                    pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

                elif model_class == 'GCN':
                    # GCN only needs node features and edge_index
                    pred = model(batch.x, batch.edge_index)

                else:
                    pred = model(batch.x, batch.edge_attr, batch.edge_index)

                loss = loss_fn(pred, batch.y)
        else:
            # No AMP, regular forward pass
            if model_class == 'BSMS_MeshGraphNet':
                # BSMS model needs multi_data dict
                multi_data = {}
                for key, value in batch.multi_data.items():
                    if isinstance(value, list):
                        multi_data[key] = [v.to(device) if torch.is_tensor(v) else v for v in value]
                    else:
                        multi_data[key] = value.to(device) if torch.is_tensor(value) else value
                pred = model(batch.x, batch.edge_attr, batch.edge_index, multi_data)

            elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

            elif model_class in ['MLPNet']:
                pred = model(batch.x)

            elif model_class == 'TransolverAero':
                # Transolver uses batch tensor for PyG batching
                pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

            elif model_class == 'GCN':
                # GCN only needs node features and edge_index
                pred = model(batch.x, batch.edge_index)

            else:
                pred = model(batch.x, batch.edge_attr, batch.edge_index)

            loss = loss_fn(pred, batch.y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.detach()

        # Step profiler if active
        if profiler is not None:
            profiler.step()

    return total_loss.sum().item() / len(loader)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp=False, amp_dtype=None, profiler=None):
    """Evaluate the model.

    Args:
        model: The model to evaluate
        loader: DataLoader for evaluation data
        loss_fn: Loss function
        device: Device to use
        use_amp: Whether to use automatic mixed precision
        amp_dtype: Data type for AMP (e.g., torch.bfloat16)
        profiler: Optional PyTorch profiler instance
    """
    model.eval()
    total_loss = 0.0

    # Determine device type for autocast
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'

    for batch in loader:
        batch = batch.to(device)

        # Check if model needs batch parameter (for poolMGN and MeshGraphNet_v2)
        model_class = model.__class__.__name__

        # Use autocast if AMP is enabled
        if use_amp and amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                if model_class == 'BSMS_MeshGraphNet':
                    # BSMS model needs multi_data dict
                    multi_data = {}
                    for key, value in batch.multi_data.items():
                        if isinstance(value, list):
                            multi_data[key] = [v.to(device) if torch.is_tensor(v) else v for v in value]
                        else:
                            multi_data[key] = value.to(device) if torch.is_tensor(value) else value
                    pred = model(batch.x, batch.edge_attr, batch.edge_index, multi_data)

                elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                    pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
                elif model_class in ['MLPNet']:
                    pred = model(batch.x)

                elif model_class == 'TransolverAero':
                    # Transolver uses batch tensor for PyG batching
                    pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

                elif model_class == 'GCN':
                    # GCN only needs node features and edge_index
                    pred = model(batch.x, batch.edge_index)

                else:
                    pred = model(batch.x, batch.edge_attr, batch.edge_index)

                loss = loss_fn(pred, batch.y)
        else:
            # No AMP, regular forward pass
            if model_class == 'BSMS_MeshGraphNet':
                # BSMS model needs multi_data dict
                multi_data = {}
                for key, value in batch.multi_data.items():
                    if isinstance(value, list):
                        multi_data[key] = [v.to(device) if torch.is_tensor(v) else v for v in value]
                    else:
                        multi_data[key] = value.to(device) if torch.is_tensor(value) else value
                pred = model(batch.x, batch.edge_attr, batch.edge_index, multi_data)

            elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
            elif model_class in ['MLPNet']:
                pred = model(batch.x)

            elif model_class == 'TransolverAero':
                # Transolver uses batch tensor for PyG batching
                pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

            elif model_class == 'GCN':
                # GCN only needs node features and edge_index
                pred = model(batch.x, batch.edge_index)

            else:
                pred = model(batch.x, batch.edge_attr, batch.edge_index)

            loss = loss_fn(pred, batch.y)

        total_loss += loss.detach()

        # Step profiler if active
        if profiler is not None:
            profiler.step()

    return total_loss.sum().item() / len(loader)

def create_model(model_config: dict, 
                 input_node_dim: int, 
                 input_edge_dim: int,
                 output_node_dim: int,
                 pos_dim: int = 2) -> 'torch.nn.Module':
    """
    Factory function to create a model based on the model name and configuration.

    Args:
        model_name (str): Name of the model to instantiate
        model_config (dict): Model configuration parameters
        input_node_dim (int): Dimension of input node features
        input_edge_dim (int): Dimension of input edge features
        output_node_dim (int): Dimension of output predictions
        pos_dim (int): Dimension of position features (2 for 2D, 3 for 3D). Default: 2

    Returns:
        torch.nn.Module: The instantiated model

    Raises:
        ValueError: If model_name is not recognized
    """
    
    model_name = model_config.get('name')


    if model_name == 'MLP' or model_name == 'mlpnet':
        from models.mlpnet import MLPNet
        model = MLPNet(
            input_node_dim=input_node_dim,
            output_node_dim=output_node_dim,
            hidden_dim=model_config.get('hidden_dim'),
            num_hidden_layers_encoder=model_config.get('num_hidden_layers_encoder'),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder'),
            activation_fn=model_config.get('activation'),
            dropout=model_config.get('dropout')
        )

    elif model_name == "meshgraphnet":
        from models.mgn import MeshGraphNet
        model = MeshGraphNet(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            output_node_dim=output_node_dim,
            processor_size=model_config.get('processor_size'),
            activation_fn=model_config.get('activation_fn'),
            num_hidden_layers_node_processor=model_config.get('num_hidden_layers_node_processor'),
            num_hidden_layers_edge_processor=model_config.get('num_hidden_layers_edge_processor'),
            hidden_dim_processor=model_config.get('hidden_dim'),
            num_hidden_layers_node_encoder=model_config.get('num_hidden_layers_node_encoder'),
            hidden_dim_node_encoder=model_config.get('hidden_dim'),
            num_hidden_layers_edge_encoder=model_config.get('num_hidden_layers_edge_encoder'),
            hidden_dim_edge_encoder=model_config.get('hidden_dim'),
            aggregation=model_config.get('aggregation'),
            hidden_dim_decoder=model_config.get('hidden_dim'),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder'),
            do_concat_trick=model_config.get('do_concat_trick')
        )

    elif model_name == 'poolMGN':
        from models.poolmgn import poolMGN
        model = poolMGN(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            output_node_dim=output_node_dim,
            processor_size=model_config.get('processor_size'),
            activation_fn=model_config.get('activation_fn'),
            num_hidden_layers_node_processor=model_config.get('num_hidden_layers_node_processor'),
            num_hidden_layers_edge_processor=model_config.get('num_hidden_layers_edge_processor'),
            hidden_dim_processor=model_config.get('hidden_dim'),
            num_hidden_layers_node_encoder=model_config.get('num_hidden_layers_node_encoder'),
            hidden_dim_node_encoder=model_config.get('hidden_dim'),
            num_hidden_layers_edge_encoder=model_config.get('num_hidden_layers_edge_encoder'),
            hidden_dim_edge_encoder=model_config.get('hidden_dim'),
            aggregation=model_config.get('aggregation'),
            hidden_dim_decoder=model_config.get('hidden_dim'),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder'),
            global_pool_method=model_config.get('global_pool_method'),
            num_hidden_layers_global_encoder=model_config.get('num_hidden_layers_global_encoder'),
            global_dim=model_config.get('global_dim'),
            dropout=model_config.get('dropout')
        )

    elif model_name == "fouriermgn":
        from models.fouriermgn import FourierMeshGraphNet
        model = FourierMeshGraphNet(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            output_node_dim=output_node_dim,
            processor_size=model_config.get('processor_size'),
            activation_fn=model_config.get('activation_fn'),
            num_hidden_layers_node_processor=model_config.get('num_hidden_layers_node_processor'),
            num_hidden_layers_edge_processor=model_config.get('num_hidden_layers_edge_processor'),
            hidden_dim_processor=model_config.get('hidden_dim'),
            num_hidden_layers_node_encoder=model_config.get('num_hidden_layers_node_encoder'),
            hidden_dim_node_encoder=model_config.get('hidden_dim'),
            num_hidden_layers_edge_encoder=model_config.get('num_hidden_layers_edge_encoder'),
            hidden_dim_edge_encoder=model_config.get('hidden_dim'),
            aggregation=model_config.get('aggregation'),
            hidden_dim_decoder=model_config.get('hidden_dim'),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder'),
            dropout=model_config.get('dropout'),
            fourier_features_dim=model_config.get('fourier_features_dim'),
            fourier_freq_start=model_config.get('fourier_freq_start'),
            fourier_freq_length=model_config.get('fourier_freq_length')
        )

    elif model_name == 'trial1' or model_name == 'Trial1':
        from models.trial1 import MeshGraphNet_v2
        model = MeshGraphNet_v2(
            node_input_size=input_node_dim,
            edge_input_size=input_edge_dim,
            hidden_channels=model_config.get('hidden_dim'),
            out_channels=output_node_dim,
            num_graph_conv_layers=model_config.get('num_message_passing_layers'),
            num_encoder_layers=model_config.get('number_of_encoding_layers'),
            num_decoder_layers=model_config.get('number_of_decoding_layers'),
            dropout=model_config.get('dropout')
        )

    elif model_name == 'bsms_mgn':
        from models.bsms_mgn import BSMS_MeshGraphNet
        model = BSMS_MeshGraphNet(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            output_node_dim=output_node_dim,
            num_levels=model_config.get('num_levels', 3),
            latent_dim=model_config.get('hidden_dim', 128),
            hidden_dim=model_config.get('hidden_dim', 128),
            pos_dim=pos_dim,
            num_hidden_layers_encoder=model_config.get('num_hidden_layers_encoder', 2),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder', 2),
            activation_fn=model_config.get('activation_fn', 'relu'),
            dropout=model_config.get('dropout', 0.0)
        )
    
    elif model_name == 'weightedgraphnet':
        from models.weightedGraphNet import WeightedGraphNet
        model = WeightedGraphNet(
            input_node_dim=input_node_dim,
            input_edge_dim=input_edge_dim,
            output_node_dim=output_node_dim,
            processor_size=model_config.get('processor_size'),
            activation_fn=model_config.get('activation_fn'),
            num_hidden_layers_edge_weight=model_config.get('num_hidden_layers_edge_weight'),
            num_hidden_layers_node_update=model_config.get('num_hidden_layers_node_update'),
            hidden_dim_processor=model_config.get('hidden_dim'),
            num_hidden_layers_node_encoder=model_config.get('num_hidden_layers_node_encoder'),
            hidden_dim_node_encoder=model_config.get('hidden_dim'),
            num_hidden_layers_edge_encoder=model_config.get('num_hidden_layers_edge_encoder'),
            hidden_dim_edge_encoder=model_config.get('hidden_dim'),
            aggregation=model_config.get('aggregation'),
            hidden_dim_decoder=model_config.get('hidden_dim'),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder'),
            dropout=model_config.get('dropout')
        )

    elif model_name == 'transolver':
        from models.transolver import TransolverAero
        model = TransolverAero(
            input_node_dim=input_node_dim,
            output_node_dim=output_node_dim,
            space_dim=pos_dim,  # 2 for 2D, 3 for 3D
            n_layers=model_config.get('n_layers', 6),
            n_hidden=model_config.get('n_hidden', 256),
            dropout=model_config.get('dropout', 0.0),
            n_head=model_config.get('n_head', 8),
            act=model_config.get('act', 'gelu'),
            mlp_ratio=model_config.get('mlp_ratio', 4),
            slice_num=model_config.get('slice_num', 32),
            use_checkpoint=model_config.get('use_checkpoint', True),
            fourier_features=model_config.get('fourier_features', False),
            fourier_dim=model_config.get('fourier_dim', 0),
            condition_dim=model_config.get('condition_dim', 0),
        )

    elif model_name == 'gcn':
        from models.gcn import GCN
        model = GCN(
            input_node_dim=input_node_dim,
            output_node_dim=output_node_dim,
            processor_size=model_config.get('processor_size', 15),
            activation_fn=model_config.get('activation_fn', 'relu'),
            hidden_dim_processor=model_config.get('hidden_dim', 128),
            num_hidden_layers_node_encoder=model_config.get('num_hidden_layers_node_encoder', 1),
            hidden_dim_node_encoder=model_config.get('hidden_dim', 128),
            hidden_dim_decoder=model_config.get('hidden_dim', 128),
            num_hidden_layers_decoder=model_config.get('num_hidden_layers_decoder', 1),
            dropout=model_config.get('dropout', 0.0)
        )

    else:
        available_models = ['MLP', 'mlpnet', 'meshgraphnet', 'poolMGN', 'fouriermgn', 'trial1', 'Trial1', 'bsms_mgn', 'weightedgraphnet', 'transolver', 'gcn']
        raise ValueError(
            f"Unknown model type: '{model_name}'. "
            f"Available models: {', '.join(available_models)}"
        )

    return model

def create_optimizer(model, training_config):
    """
    Factory function to create an optimizer based on config.

    Args:
        model: The model whose parameters will be optimized
        training_config (dict): Training configuration with optimizer parameters

    Returns:
        torch.optim.Optimizer: The instantiated optimizer

    Raises:
        ValueError: If optimizer type is not recognized
    """
    optimizer_type = training_config.get('optimizer')
    learning_rate = training_config.get('learning_rate')
    weight_decay = training_config.get('weight_decay')

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            fused=True
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_type}. "
            f"Supported types: 'Adam', 'AdamW'"
        )

    return optimizer

def create_scheduler(optimizer, training_config):
    """
    Factory function to create a learning rate scheduler based on config.

    Args:
        optimizer: The optimizer to apply scheduling to
        training_config (dict): Training configuration with scheduler parameters

    Returns:
        torch.optim.lr_scheduler: The instantiated scheduler

    Raises:
        ValueError: If scheduler type is not recognized
    """
    scheduler_type = training_config.get('scheduler')

    if scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=training_config.get('lr_scheduler_exp_gamma')
        )
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('epochs'),
            eta_min=1e-7
        )
    elif scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=training_config.get('lr_scheduler_T_0'),
            T_mult=1,
            eta_min=1e-7
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config.get('lr_scheduler_gamma'),
            patience=training_config.get('lr_scheduler_step_size'),
            min_lr=1e-7
        )
    elif scheduler_type == "Warmup":
        scheduler = WarmupCosineDecayScheduler(
            optimizer,
            warmup=training_config.get('warmup_steps', 100),
            max_iters=training_config.get('epochs')
        )
    else:
        raise ValueError(
            f"Unknown scheduler type: {scheduler_type}. "
            f"Supported types: 'ExponentialLR', 'ReduceLROnPlateau', "
            f"'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'Warmup'"
        )

    return scheduler

class WarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = epoch * 1.0 / self.warmup
        else:
            progress = (epoch - self.warmup) / (self.max_num_iters - self.warmup)
            lr_factor = 0.5 * (1 + np.cos(np.pi * progress))
        return lr_factor
