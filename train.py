import datetime
import os
import yaml
import torch
from torch_geometric.loader import DataLoader
from torch import nn
import argparse
from utils import get_experiment_config, train, evaluate
from torch_geometric.data import Dataset
from dataset import create_datasets
import glob
from pathlib import Path
from tqdm import tqdm
import random
from collections import defaultdict

#%%Main
def main(params):
    # Set up precision (float32 or float64)
    # For bfloat16, we use automatic mixed precision (AMP) instead of casting data
    precision = params['training'].get('precision').lower()
    use_amp = False  # Flag for automatic mixed precision
    amp_dtype = None  # AMP dtype for autocast

    if precision in ['double', 'float64']:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
        print("Using double precision (float64)")
    elif precision in ['float', 'float32', 'single']:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
        print("Using single precision (float32)")

    elif precision in ['bf16', 'bfloat16']:
        # For bfloat16, keep data in float32 and use AMP for operations
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
        use_amp = True
        amp_dtype = torch.bfloat16
        print("Using automatic mixed precision with bfloat16")
        print("  - Data: float32 (preserves precision)")
        print("  - Operations: bfloat16 (faster computation)")

    else:
        raise ValueError(f"Unknown precision type: {precision}. Supported types: 'float32', 'float64', 'bfloat16', 'single'")
    
    #instantiate dataset
    train_set, val_set, test_set, norm_stats = create_datasets(
        data_dir=params['dataset']['data_dir'],
        dataset_type=params['dataset']['name'],
        params=params,
        dtype=dtype
    )

    # Check if using BSMS model which requires special data preprocessing
    model_name = params['model']['name']
    if model_name == 'bsms_mgn':
        from models.bsms_dataset_wrapper import BSMSDataLoader, prepare_bsms_data

        print("\n=== Preprocessing Multi-Scale Graphs for BSMS ===")
        num_levels = params['model'].get('num_levels', 3)

        train_set = prepare_bsms_data(train_set, num_levels=num_levels)
        val_set = prepare_bsms_data(val_set, num_levels=num_levels)
        test_set = prepare_bsms_data(test_set, num_levels=num_levels)

        train_loader = BSMSDataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = BSMSDataLoader(val_set, batch_size=1, shuffle=False)
    else:
        train_loader = DataLoader(train_set, batch_size=params['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_set, batch_size=params['training']['batch_size'])


    device = torch.device(params["training"]["device"] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get feature dimensions from the first data sample
    sample_batch = next(iter(train_loader))
    input_node_dim = sample_batch.x.shape[1]
    input_edge_dim = sample_batch.edge_attr.shape[1]
    output_node_dim = sample_batch.y.shape[1]
    pos_dim = sample_batch.pos.shape[1] if hasattr(sample_batch, 'pos') else 2

    # print(f"Input node features: {input_node_dim}")
    # print(f"Input edge features: {input_edge_dim}")
    # print(f"Output dimension: {output_node_dim}")

    # Model instantiation based on configuration (model_name already defined above)
    model_config = params['model']
    
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
            do_concat_trick=model_config.get('do_concat_trick'),
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

    else:
        raise ValueError(f"Unknown model type: {model_name}. Available models: 'MLP', 'meshgraphnet', 'poolMGN', 'fouriermgn', 'trial1', 'bsms_mgn'")
        
    print(model)

    # Convert model to correct precision
    # Note: For bfloat16 with AMP, we keep the model in float32
    if dtype == torch.float64:
        model = model.double()
        print("Model converted to double precision")

    model = model.to(device)
    print(f"Model moved to {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training configuration
    training_config = params['training']
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=training_config.get('learning_rate'),
        
        weight_decay=training_config.get('weight_decay')
    )
    

    if training_config.get('scheduler') == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=training_config.get('lr_scheduler_exp_gamma')
        )
    elif training_config.get('scheduler') == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_config.get('epochs'),
            eta_min=1e-7
        )
    elif training_config.get('scheduler') == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=training_config.get('lr_scheduler_T_0'),
            T_mult=1,
            eta_min=1e-7
        )
    elif training_config.get('scheduler') == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=training_config.get('lr_scheduler_gamma'),
            patience=training_config.get('lr_scheduler_step_size'),
            min_lr=1e-7
        )
    elif training_config.get('scheduler') == "Warmup":
        from utils import WarmupCosineDecayScheduler
        scheduler = WarmupCosineDecayScheduler(
            optimizer,
            warmup=training_config.get('warmup_steps', 100),
            max_iters=training_config.get('epochs')
        )
    else:
        raise ValueError(f"Unknown scheduler type: {training_config.get('scheduler')}. Supported types: 'ExponentialLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts'")  
    
    loss_fn = nn.MSELoss()

    # Training loop
    
    if training_config.get('epochs', 0) > 0:
        
        train_losses = []
        val_losses = []
        lr_list = []
        iterator = tqdm(range(training_config['epochs']))
        val_loss_min = float('inf')
        patience_counter = 0
        
        for epoch in iterator:

            train_loss = train(model, train_loader, optimizer, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype)
            val_loss = evaluate(model, val_loader, loss_fn, device, use_amp=use_amp, amp_dtype=amp_dtype)
            if training_config.get('scheduler') == 'ReduceLROnPlateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            iterator.set_postfix(Loss=train_loss, Val_Loss=val_loss, lr=current_lr)
            
            # Early stopping
            if training_config.get('early_stopping'):
                if val_loss < val_loss_min:
                    val_loss_min = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter > training_config.get('patience'):
                        print(f"\nEarly stopping at epoch {epoch}")
                        break
                        
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            lr_list.append(current_lr)
        
        print("\nTraining complete.")
        
        # Create output directory with date folder structure
        now = datetime.datetime.now()
        date_folder = now.strftime("%d-%m-%Y")  # e.g., "22-09-2024"
        time_stamp = now.strftime("%H-%M")      # e.g., "14-30"
        
        model_info = f"{model_name}-{params['dataset']['name']}"
        
        # Create: training_runs/22-09-2024/14-30-model-dataset
        run_folder = f"{time_stamp}-{model_info}"
        save_dir = os.path.join("training_runs", date_folder, run_folder)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model state dict
        torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))
        # print(f"Model weights saved to {save_dir}/model_weights.pt")
        
        # Save normalization statistics
        torch.save(norm_stats, os.path.join(save_dir, "normalization_stats.pt"))
        # print(f"Normalization stats saved to {save_dir}/normalization_stats.pt")
        
        # Save training configuration and parameters
        import json
        params_to_save = {}
        for key, value in params.items():
            try:
                json.dumps(value)  # Test if value is JSON serializable
                params_to_save[key] = value
            except (TypeError, ValueError):
                params_to_save[key] = str(value)  # Convert non-serializable to string
        
        with open(os.path.join(save_dir, "experiment_params.json"), "w") as f:
            json.dump(params_to_save, f, indent=2)
        # print(f"Parameters saved to {save_dir}/experiment_params.json")
        
        # Save loss history
        loss_data = {
            'final_train_loss': train_losses[-1] if train_losses else 0.0,
            'final_val_loss': val_losses[-1] if val_losses else 0.0,
            'best_val_loss': min(val_losses) if val_losses else 0.0,
            'final_lr': lr_list[-1] if lr_list else 0.0,
            'total_epochs': len(train_losses),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'learning_rates': lr_list
        }
        
        with open(os.path.join(save_dir, "training_losses.json"), "w") as f:
            json.dump(loss_data, f, indent=2)
        
        # Create high-resolution loss plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, label='Training Loss', linewidth=2, color='blue')
        plt.plot(val_losses, label='Validation Loss', linewidth=2, color='red')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss (MSE)', fontsize=14)
        plt.title(f'Training Progress - {model_name}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.yscale('log') 
        
        # Add best validation loss annotation
        best_epoch = val_losses.index(min(val_losses))
        plt.annotate(f'Best Val Loss: {min(val_losses):.6f}\nEpoch: {best_epoch}',
                    xy=(best_epoch, min(val_losses)), xytext=(best_epoch + len(val_losses)*0.1, min(val_losses)*2),
                    arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_loss_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        
        plt.figure(figsize=(12, 8))
        plt.plot(lr_list, label='Learning Rate', linewidth=2, color='green')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        scheduler_type = training_config.get('scheduler')
        plt.title(f'Learning Rate Schedule - {scheduler_type}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "learning_rate_schedule.png"), dpi=300, bbox_inches='tight')
        plt.close() 
        
        # Save summary text file
        with open(os.path.join(save_dir, "training_summary.txt"), "w") as f:
            f.write(f"Training Summary - {date_folder} {time_stamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Experiment: {params.get('experiment_name', 'Unknown')}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {params['dataset']['name']}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  Hidden Dim: {model_config.get('hidden_dim')}\n")
            if model_name in ['meshgraphnet', 'MGN', 'initialglobalmgn', 'trial1']:
                f.write(f"  Message Passing Layers: {model_config.get('num_message_passing_layers')}\n")
                
                if model_name in ['initialglobalmgn', 'initialglobal']:
                    f.write(f"  Initial Global Pool Type: {model_config.get('global_pool_type', 'mean')} (once at start)\n")
                    f.write(f"  Global Dim: {model_config.get('global_dim', 'same as hidden_dim')}\n")
                elif model_name in ['separateddecoder']:
                    f.write(f"  Decoder Architecture: Separate decoders for P, tau, T\n")
            f.write(f"  Input Node Features: {input_node_dim}\n")
            f.write(f"  Input Edge Features: {input_edge_dim}\n")
            f.write(f"  Output Features: {output_node_dim}\n\n")
            
            f.write(f"Normalization Statistics:\n")
            f.write(f"  Node features - Mean: {norm_stats.get('node_mean')}\n")
            f.write(f"  Node features - Std: {norm_stats.get('node_std')}\n")
            f.write(f"  Edge attributes - Mean: {norm_stats.get('edge_mean')}\n")
            f.write(f"  Edge attributes - Std: {norm_stats.get('edge_std')}\n")
            f.write(f"  Targets - Mean: {norm_stats.get('target_mean')}\n")
            f.write(f"  Targets - Std: {norm_stats.get('target_std')}\n\n")

            f.write("Training Configuration:\n")
            f.write(f"  Learning Rate: {training_config.get('learning_rate', 0.001)}\n")
            f.write(f"  Weight Decay: {training_config.get('weight_decay', 1e-5)}\n")
            f.write(f"  Batch Size: {params['training']['batch_size']}\n")
            f.write(f"  Early Stopping: {training_config.get('early_stopping', False)}\n")
            if training_config.get('early_stopping', False):
                f.write(f"  Patience: {training_config.get('patience', 50)}\n\n")
            
            f.write("Training Results:\n")
            f.write(f"  Total Epochs: {len(train_losses)}\n")
            f.write(f"  Final Training Loss: {train_losses[-1]:.6f}\n")
            f.write(f"  Final Validation Loss: {val_losses[-1]:.6f}\n")
            f.write(f"  Best Validation Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses))})\n")
            
            if training_config.get('early_stopping') and patience_counter > training_config.get('patience'):
                f.write(f"  Training stopped early due to no improvement for {training_config.get('patience')} epochs\n")
                
            #if dataset is airfoil, write train, validation, test airfoil names.
            f.write(f"Dataset Splits:\n")
            if params['dataset']['name'] == 'airfoil_2d':
                # Extract unique airfoil names from the data lists
                train_airfoil_names = list(set(data.airfoil for data in train_set if hasattr(data, 'airfoil')))
                val_airfoil_names = list(set(data.airfoil for data in val_set if hasattr(data, 'airfoil')))
                test_airfoil_names = list(set(data.airfoil for data in test_set if hasattr(data, 'airfoil')))
                
                f.write(f"  Train airfoils: {train_airfoil_names}\n")
                f.write(f"  Validation airfoils: {val_airfoil_names}\n")
                f.write(f"  Test airfoils: {test_airfoil_names}\n\n")
            elif params['dataset']['name'] == 'ahmed_body':
                train_case_names = list(set(data.case_name for data in train_set if hasattr(data, 'case_name')))
                val_case_names = list(set(data.case_name for data in val_set if hasattr(data, 'case_name')))
                test_case_names = list(set(data.case_name for data in test_set if hasattr(data, 'case_name')))
                
                f.write(f"  Train cases: {train_case_names}\n")
                f.write(f"  Validation cases: {val_case_names}\n")
                f.write(f"  Test cases: {test_case_names}\n\n")

        print(f"Training summary saved to {save_dir}/training_summary.txt")
        print(f"\nAll outputs saved to: {save_dir}")
    
    else:
        print("Training skipped (epochs = 0)")
        save_dir = "results"
        os.makedirs(save_dir, exist_ok=True)
    
    # Inference on test set
    try:
        from inference import AeroInference
        print("Running inference on test set...")

        # Create inference engine with AMP settings
        inference_engine = AeroInference(model, norm_stats, device, params, use_amp=use_amp, amp_dtype=amp_dtype)

        # Run inference and save results
        inference_dir = inference_engine.run_inference(test_set, save_dir, params['dataset'].get('data_dir'))
        print(f"Inference results saved to {inference_dir}")
        
    except ImportError as e:
        print(f"Could not import inference module: {e}")
        print("Skipping inference step")
    except Exception as e:
        print(f"Error during inference: {e}")
        print("Continuing without inference...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GNN model')
    parser.add_argument(
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name defined in config.yaml",
    )
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    experiments = configs.get("experiments", {})
    if args.experiment not in experiments:
        available = ", ".join(sorted(experiments.keys())) or "<none>"
        raise ValueError(
            f"Experiment '{args.experiment}' not found in configuration. Available: {available}"
        )

    params = get_experiment_config(experiments[args.experiment], configs)
    params["experiment_name"] = args.experiment

    main(params)
