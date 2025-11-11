"""
Test RCM reordering performance impact on training.
"""
import time
import torch
import yaml
import argparse
from utils import get_experiment_config
from dataset import create_datasets
from train_utils import create_model, create_optimizer
from torch_geometric.loader import DataLoader
from torch import nn

def time_training_epoch(config, use_rcm=False, num_epochs=3):
    """Time a single training setup and iteration."""

    # Temporarily modify config
    original_rcm = config["training"].get("reordering", None)
    config["training"]["reordering"] = "rcm" if use_rcm else None

    # Set up precision like train.py does
    precision = config['training'].get('precision', 'float32').lower()
    use_amp = False
    amp_dtype = None

    if precision in ['double', 'float64']:
        torch.set_default_dtype(torch.float64)
        dtype = torch.float64
    elif precision in ['bf16', 'bfloat16']:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32
        use_amp = True
        amp_dtype = torch.bfloat16
    else:
        torch.set_default_dtype(torch.float32)
        dtype = torch.float32

    print(f"\n{'='*60}")
    print(f"Testing with RCM: {use_rcm}")
    print(f"{'='*60}")

    # Time data loading
    start = time.time()
    train_data, val_data, test_data, norm_stats = create_datasets(
        config["dataset"]["data_dir"],
        config["dataset"]["name"],
        config,
        dtype=dtype
    )
    load_time = time.time() - start
    print(f"Data loading time: {load_time:.2f}s")
    print(f"Train samples: {len(train_data)}")

    # Create data loader
    train_loader = DataLoader(
        train_data,
        batch_size=config["training"]["batch_size"],
        shuffle=True
    )

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Get feature dimensions from the first data sample
    sample_batch = next(iter(train_loader))
    input_node_dim = sample_batch.x.shape[1]
    input_edge_dim = sample_batch.edge_attr.shape[1]
    output_node_dim = sample_batch.y.shape[1]
    pos_dim = sample_batch.pos.shape[1] if hasattr(sample_batch, 'pos') else 2

    model_config = config['model']
    model = create_model(
        model_config=model_config,
        input_node_dim=input_node_dim,
        input_edge_dim=input_edge_dim,
        output_node_dim=output_node_dim,
        pos_dim=pos_dim
    )

    # Convert model to correct precision
    if dtype == torch.float64:
        model = model.double()

    model = model.to(device)

    training_config = params['training']
    # Create optimizer
    optimizer = create_optimizer(model, training_config)

    # Loss function
    criterion = nn.MSELoss()

    # Time actual training epochs
    print(f"\nRunning {num_epochs} training epochs...")
    

    for epoch in range(num_epochs+5):
        if epoch == 5:
            # Start timing after warm-up epochs
            start = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            model_class = model.__class__.__name__

            # Forward pass with AMP if enabled
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    if model_class == "MeshGraphNet":
                        # Special handling for MeshGraphNet
                        predictions = model(batch.x, batch.edge_attr, batch.edge_index)
                    elif model_class == "GCN":
                        predictions = model(batch.x, batch.edge_index)
                    loss = criterion(predictions, batch.y)
            else:
                if model_class == "MeshGraphNet":
                    # Special handling for MeshGraphNet
                    predictions = model(batch.x, batch.edge_attr, batch.edge_index)
                elif model_class == "GCN":
                    predictions = model(batch.x, batch.edge_index)
                loss = criterion(predictions, batch.y)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

        avg_loss = epoch_loss / batch_count
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}, Batches = {batch_count}")

    training_time = time.time() - start
    time_per_epoch = training_time / num_epochs
    print(f"Total training time: {training_time:.2f}s")
    print(f"Average time per epoch: {time_per_epoch:.2f}s")

    # Restore config
    config["training"]["reordering"] = original_rcm

    return load_time, training_time, time_per_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RCM reordering performance')
    parser.add_argument(
        "--exp",
        "--experiment",
        dest="experiment",
        type=str,
        required=True,
        help="Experiment name defined in config.yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to run for timing (default: 3)",
    )
    args = parser.parse_args()

    # Load config like train.py does
    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)

    experiments = configs.get("experiments", {})
    if args.experiment not in experiments:
        available = ", ".join(sorted(experiments.keys())) or "<none>"
        raise ValueError(
            f"Experiment '{args.experiment}' not found in configuration. Available: {available}"
        )

    params = get_experiment_config(experiments[args.experiment], configs)

    # Number of epochs to test
    num_epochs = args.epochs
    print(f"Testing with {num_epochs} epochs per run")

    # Test without RCM
    load_no_rcm, total_no_rcm, per_epoch_no_rcm = time_training_epoch(params, use_rcm=False, num_epochs=num_epochs)

    # Test with RCM
    load_rcm, total_rcm, per_epoch_rcm = time_training_epoch(params, use_rcm=True, num_epochs=num_epochs)

    # Compare
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Data loading:")
    print(f"  Without RCM: {load_no_rcm:.2f}s")
    print(f"  With RCM:    {load_rcm:.2f}s")
    print(f"  Difference:  {load_rcm - load_no_rcm:+.2f}s ({(load_rcm/load_no_rcm-1)*100:+.1f}%)")

    print(f"\nTraining time ({num_epochs} epochs):")
    print(f"  Without RCM: {total_no_rcm:.2f}s")
    print(f"  With RCM:    {total_rcm:.2f}s")
    print(f"  Difference:  {total_rcm - total_no_rcm:+.2f}s ({(total_rcm/total_no_rcm-1)*100:+.1f}%)")

    print(f"\nAverage time per epoch:")
    print(f"  Without RCM: {per_epoch_no_rcm:.2f}s")
    print(f"  With RCM:    {per_epoch_rcm:.2f}s")
    print(f"  Difference:  {per_epoch_rcm - per_epoch_no_rcm:+.2f}s ({(per_epoch_rcm/per_epoch_no_rcm-1)*100:+.1f}%)")

    print(f"\nTotal time (loading + training):")
    total_time_no_rcm = load_no_rcm + total_no_rcm
    total_time_rcm = load_rcm + total_rcm
    print(f"  Without RCM: {total_time_no_rcm:.2f}s")
    print(f"  With RCM:    {total_time_rcm:.2f}s")
    print(f"  Difference:  {total_time_rcm - total_time_no_rcm:+.2f}s ({(total_time_rcm/total_time_no_rcm-1)*100:+.1f}%)")
