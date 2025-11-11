"""
Profiling script for GNN models (MeshGraphNet, GCN, etc.)
Works with the config system like train.py

Usage:
    python profile_mgn_example.py --exp airfoil_mgn
    python profile_mgn_example.py --exp airfoil_gcn
    python profile_mgn_example.py --exp ahmedBody_gcn
"""

import torch
import torch.nn as nn
import argparse
from torch_geometric.loader import DataLoader
from models.profiling_utils import enable_profiling, disable_profiling, print_profiling_summary, reset_profiling, get_profiling_summary
from utils import get_experiment_config
from dataset import create_datasets
from train_utils import create_model
import yaml


def profile_with_real_data(exp_name: str, num_iterations: int = 10, num_samples: int = 5):
    """
    Profile a model using real dataset with experiment configuration.

    Args:
        exp_name: Name of experiment from config.yaml (e.g., 'airfoil_mgn', 'airfoil_gcn')
        num_iterations: Number of forward passes per sample
        num_samples: Number of dataset samples to profile
    """
    print("\n" + "="*80)
    print(f"Profiling experiment: {exp_name}")
    print("="*80)
    
    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    experiments = configs.get("experiments", {})
    if exp_name not in experiments:
        available = ", ".join(sorted(experiments.keys())) or "<none>"
        raise ValueError(
            f"Experiment '{args.experiment}' not found in configuration. Available: {available}"
        )

    params = get_experiment_config(experiments[exp_name], configs)
    params["experiment_name"] = exp_name

    # Load experiment config (like train.py)
    # params = get_experiment_config(exp_name)
    model_name = params['model']['name']
    dataset_name = params['dataset']['name']

    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_name}")

    # Set up precision
    precision = params['training'].get('precision', 'single').lower()
    if precision in ['double', 'float64']:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    elif precision in ['bfloat16', 'bf16']:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32

    # Create datasets (like train.py)
    print("\nLoading dataset...")
    train_set, val_set, test_set, norm_stats = create_datasets(
        data_dir=params['dataset']['data_dir'],
        dataset_type=params['dataset']['name'],
        params=params,
        dtype=dtype
    )

    print(f"Train samples: {len(train_set)}")
    print(f"Using {min(num_samples, len(train_set))} samples for profiling")

    # Create data loader
    batch_size = params['training'].get('batch_size', 1)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # Get sample batch to determine dimensions
    sample_batch = next(iter(train_loader))
    input_node_dim = sample_batch.x.shape[1]
    input_edge_dim = sample_batch.edge_attr.shape[1]
    output_node_dim = sample_batch.y.shape[1]
    pos_dim = sample_batch.pos.shape[1] if hasattr(sample_batch, 'pos') else 2

    print(f"\nInput node dim: {input_node_dim}")
    print(f"Input edge dim: {input_edge_dim}")
    print(f"Output node dim: {output_node_dim}")
    print(f"Spatial dim: {pos_dim}")

    # Create model using factory (like train.py)
    print(f"\nCreating {model_name} model...")
    model_config = params['model']
    model = create_model(
        model_config=model_config,
        input_node_dim=input_node_dim,
        input_edge_dim=input_edge_dim,
        output_node_dim=output_node_dim,
        pos_dim=pos_dim
    )

    # Move to device
    device = torch.device(params["training"]["device"] if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Device: {device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Warm-up (not profiled)
    print("\nWarming up...")
    model.eval()
    disable_profiling()

    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i >= 5:  # Just 5 warmup iterations
                break
            batch = batch.to(device)

            # Get model class for forward pass signature
            model_class = model.__class__.__name__

            if model_class == 'GCN':
                _ = model(batch.x, batch.edge_index)
            elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                _ = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
            elif model_class == 'TransolverAero':
                _ = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
            elif model_class in ['MLPNet']:
                _ = model(batch.x)
            else:
                _ = model(batch.x, batch.edge_attr, batch.edge_index)

    # Profile forward passes
    print(f"\nProfiling {num_iterations} iterations on {num_samples} samples...")
    enable_profiling()
    reset_profiling()

    model.eval()
    with torch.no_grad():
        for sample_idx, batch in enumerate(train_loader):
            if sample_idx >= num_samples:
                break

            batch = batch.to(device)
            model_class = model.__class__.__name__

            # Run multiple iterations on this batch
            for _ in range(num_iterations):
                if model_class == 'GCN':
                    output = model(batch.x, batch.edge_index)
                elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                    output = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
                elif model_class == 'TransolverAero':
                    output = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
                elif model_class in ['MLPNet']:
                    output = model(batch.x)
                else:
                    output = model(batch.x, batch.edge_attr, batch.edge_index)

    # Print results
    print("\n" + "="*80)
    print("PROFILING RESULTS")
    print("="*80)
    print_profiling_summary(sort_by='time_total_ms')

    disable_profiling()

    return model


def compare_models(exp_names: list, num_iterations: int = 10):
    """
    Compare profiling results across multiple models.

    Args:
        exp_names: List of experiment names to compare (e.g., ['airfoil_mgn', 'airfoil_gcn'])
        num_iterations: Number of forward passes per sample
    """
    print("\n" + "="*80)
    print("COMPARING MODELS")
    print("="*80)

    results = {}

    for exp_name in exp_names:
        print(f"\n{'='*40}")
        print(f"Profiling: {exp_name}")
        print(f"{'='*40}")

        try:
            # Profile this model
            reset_profiling()
            model = profile_with_real_data(exp_name, num_iterations=num_iterations, num_samples=3)

            # Get summary
            summary = get_profiling_summary()
            results[exp_name] = summary

        except Exception as e:
            print(f"Error profiling {exp_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    if len(results) > 1:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)

        # Find common operations
        all_ops = set()
        for summary in results.values():
            all_ops.update(summary.keys())

        # Print header
        print(f"\n{'Operation':<30}", end='')
        for exp_name in exp_names:
            print(f"{exp_name[:20]:>20}", end='')
        print()
        print("-" * (30 + 20 * len(exp_names)))

        # Print each operation
        for op in sorted(all_ops):
            print(f"{op:<30}", end='')
            for exp_name in exp_names:
                if exp_name in results and op in results[exp_name]:
                    time_ms = results[exp_name][op]['time_total_ms']
                    print(f"{time_ms:>18.2f}ms", end='')
                else:
                    print(f"{'N/A':>20}", end='')
            print()


def profile_single_forward_pass():
    """
    Example: Profile a single forward pass with synthetic data.
    """
    print("\n" + "="*80)
    print("Example: Profiling single forward pass with synthetic data")
    print("="*80)

    # Create a simple graph
    num_nodes = 1000
    num_edges = 5000
    node_dim = 16
    edge_dim = 16

    # Random data
    x = torch.randn(num_nodes, node_dim).cuda()
    edge_attr = torch.randn(num_edges, edge_dim).cuda()
    edge_index = torch.randint(0, num_nodes, (2, num_edges)).cuda()

    # Create model with proper initialization
    from models.mgn import MeshGraphNet
    model = MeshGraphNet(
        input_node_dim=node_dim,
        input_edge_dim=edge_dim,
        output_node_dim=3,
        processor_size=5,
        activation_fn='relu',
        num_hidden_layers_node_processor=2,
        num_hidden_layers_edge_processor=2,
        hidden_dim_processor=128,
        num_hidden_layers_node_encoder=2,
        hidden_dim_node_encoder=128,
        num_hidden_layers_edge_encoder=2,
        hidden_dim_edge_encoder=128,
        aggregation='add',
        hidden_dim_decoder=128,
        num_hidden_layers_decoder=2,
        dropout=0.0,
        do_concat_trick=False
    ).cuda()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Warm-up (not profiled)
    print("Warming up...")
    disable_profiling()
    with torch.no_grad():
        for _ in range(3):
            _ = model(x, edge_attr, edge_index)

    # Profile multiple forward passes
    print(f"\nProfiling {10} forward passes...")
    enable_profiling()
    reset_profiling()
    num_iterations = 10

    with torch.no_grad():
        for i in range(num_iterations):
            output = model(x, edge_attr, edge_index)

    # Print results
    print_profiling_summary(sort_by='time_total_ms')

    disable_profiling()


def profile_different_graph_sizes():
    """
    Example: Profile performance across different graph sizes.
    """
    print("\n" + "="*80)
    print("Example: Profiling different graph sizes")
    print("="*80)

    node_counts = [500, 1000, 2000, 5000]
    edge_dim = 16
    node_dim = 16

    # Create model with proper initialization
    from models.mgn import MeshGraphNet
    model = MeshGraphNet(
        input_node_dim=node_dim,
        input_edge_dim=edge_dim,
        output_node_dim=3,
        processor_size=3,
        activation_fn='relu',
        num_hidden_layers_node_processor=2,
        num_hidden_layers_edge_processor=2,
        hidden_dim_processor=128,
        num_hidden_layers_node_encoder=2,
        hidden_dim_node_encoder=128,
        num_hidden_layers_edge_encoder=2,
        hidden_dim_edge_encoder=128,
        aggregation='add',
        hidden_dim_decoder=128,
        num_hidden_layers_decoder=2,
        dropout=0.0,
        do_concat_trick=False
    ).cuda()

    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    for num_nodes in node_counts:
        num_edges = num_nodes * 5  # Roughly 5 edges per node

        print(f"\n--- Graph size: {num_nodes} nodes, {num_edges} edges ---")

        x = torch.randn(num_nodes, node_dim).cuda()
        edge_attr = torch.randn(num_edges, edge_dim).cuda()
        edge_index = torch.randint(0, num_nodes, (2, num_edges)).cuda()

        # Warm-up for this size
        disable_profiling()
        with torch.no_grad():
            for _ in range(2):
                _ = model(x, edge_attr, edge_index)

        # Enable profiling for this size
        enable_profiling()
        reset_profiling()

        # Run forward passes
        with torch.no_grad():
            for _ in range(5):
                _ = model(x, edge_attr, edge_index)

        # Print compact summary
        summary = get_profiling_summary()

        # Show only total times for key operations
        key_ops = ['total_edge_block', 'total_node_block', 'gcn_message_passing']
        print(f"  {'Operation':<30} {'Total (ms)':>15} {'Mean (ms)':>15}")
        print(f"  {'-'*30} {'-'*15} {'-'*15}")
        for op in sorted(summary.keys()):
            if any(key in op for key in ['edge', 'node', 'gcn', 'encoder', 'decoder']):
                total_ms = summary[op]['time_total_ms']
                mean_ms = summary[op]['time_mean_ms']
                print(f"  {op:<30} {total_ms:>15.3f} {mean_ms:>15.3f}")

        disable_profiling()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profile GNN models')
    parser.add_argument('--exp', type=str, default=None,
                       help='Experiment name from config.yaml')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                       help='Compare multiple experiments (e.g., --compare airfoil_mgn airfoil_gcn)')
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of iterations per sample')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of samples to profile')
    parser.add_argument('--synthetic', action='store_true',
                       help='Run synthetic data examples')

    args = parser.parse_args()

    if args.compare:
        # Compare multiple models
        compare_models(args.compare, num_iterations=args.iterations)

    elif args.exp:
        # Profile single experiment
        profile_with_real_data(
            args.exp,
            num_iterations=args.iterations,
            num_samples=args.samples
        )

    elif args.synthetic:
        # Run synthetic examples
        print("Running synthetic examples...")
        profile_single_forward_pass()
        profile_different_graph_sizes()

    else:
        # Show usage
        print("Usage examples:")
        print("  python profile_mgn_example.py --exp airfoil_mgn")
        print("  python profile_mgn_example.py --exp airfoil_gcn")
        print("  python profile_mgn_example.py --compare airfoil_mgn airfoil_gcn")
        print("  python profile_mgn_example.py --synthetic")
        print("\nNo arguments provided. Run with --help for options.")

    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80 + "\n")
