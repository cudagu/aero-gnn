# BSMS-GNN Implementation for Aero-GNN

This implementation provides a **Bi-Stride Multi-Scale Graph Neural Network** for mesh-based aerodynamic simulation, compatible with your existing dataset and model architecture.

## Overview

BSMS-GNN uses **bistride pooling** - a novel graph coarsening strategy that:
- Selects nodes at alternating BFS (breadth-first search) levels
- Eliminates need for manual mesh coarsening
- Avoids geometric errors from proximity-based pooling
- Implements U-Net-like encoder-decoder architecture for multi-scale processing

**Based on:** [Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network](https://arxiv.org/abs/2210.02573) (ICML 2023)

## Files

1. **`bistride_ops.py`** - Core bistride operations:
   - `BistridePooling`: Selects nodes using alternating BFS levels
   - `Unpool`: Restores features to original resolution
   - `WeightedEdgeConv`: Edge convolution with learned weights
   - `GMP`: Graph message passing layer

2. **`bsms_mgn.py`** - Main model architecture:
   - `MultiScaleGraphPreprocessor`: Creates multi-scale graph hierarchy
   - `BSMSGMP`: U-Net style multi-scale message passing
   - `BSMS_MeshGraphNet`: Complete model compatible with your setup

3. **`bsms_dataset_wrapper.py`** - Dataset utilities:
   - `BSMSDatasetWrapper`: Wraps AeroDataset with multi-scale preprocessing
   - `BSMSDataLoader`: Simple data loader for BSMS
   - `prepare_bsms_data()`: Helper function

4. **`example_bsms_train.py`** - Training example:
   - Complete training loop
   - Shows integration with your existing setup

## Quick Start

### 1. Basic Usage

```python
from dataset import AeroDataset
from models.bsms_mgn import BSMS_MeshGraphNet
from models.bsms_dataset_wrapper import prepare_bsms_data, BSMSDataLoader

# Create your dataset
dataset = AeroDataset(
    data_dir='path/to/data',
    dataset_type='airfoil_2d',
    params=config
)

# Wrap with multi-scale preprocessing
bsms_dataset = prepare_bsms_data(dataset, num_levels=3)

# Create loader
loader = BSMSDataLoader(bsms_dataset, batch_size=1, shuffle=True)

# Get sample to determine dimensions
sample = next(iter(loader))

# Create model
model = BSMS_MeshGraphNet(
    input_node_dim=sample.x.shape[1],
    input_edge_dim=sample.edge_attr.shape[1],
    output_node_dim=sample.y.shape[1],
    num_levels=3,
    latent_dim=128,
    hidden_dim=128,
    pos_dim=2  # 2 for 2D, 3 for 3D meshes
)

# Forward pass
predictions = model(
    sample.x,
    sample.edge_attr,
    sample.edge_index,
    sample.multi_data
)
```

### 2. Training

```bash
# Run the example training script
python example_bsms_train.py
```

Or integrate into your existing training code:

```python
def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for data in loader:
        # Move to device
        data = data.to(device)

        # Move multi_data to device
        multi_data = {}
        for key, value in data.multi_data.items():
            if isinstance(value, list):
                multi_data[key] = [v.to(device) if torch.is_tensor(v) else v
                                  for v in value]
            else:
                multi_data[key] = value

        # Forward pass
        pred = model(data.x, data.edge_attr, data.edge_index, multi_data)

        # Loss and optimization
        loss = loss_fn(pred, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)
```

### 3. Configuration

Update your config YAML:

```yaml
model:
  name: bsms_mgn
  hidden_dim: 128
  num_levels: 3  # Number of coarsening levels
  num_hidden_layers_encoder: 2
  num_hidden_layers_decoder: 2
  activation_fn: relu
  dropout: 0.0

training:
  batch_size: 1  # Currently only supports 1
  learning_rate: 1e-4
  epochs: 100
```

## Architecture Details

### Multi-Scale Hierarchy

The model creates a hierarchy of graphs at different scales:

```
Level 0 (Finest):  N nodes, E edges
Level 1:          N/2 nodes (bistride pooling)
Level 2:          N/4 nodes (bistride pooling)
...
```

### U-Net Architecture

```
Input (finest mesh)
    ↓ Encode
    ↓ GMP + EdgeConv
    ↓ Pool (bistride)
    ├────────────┐ Skip connection
    ↓            │
  Coarser        │
    ↓ GMP        │
    ↓ Pool       │
    ├──────┐     │
    ↓      │     │
Coarsest   │     │
    ↓ GMP  │     │
    ↓      │     │
    ↓ Unpool     │
    ↓ EdgeConv   │
    ↓ <──────────┘ Add skip
    ↓ Unpool
    ↓ EdgeConv
    ↓ <──────────────┘ Add skip
    ↓ Decode
Output predictions
```

### Bistride Pooling Algorithm

1. Select a seed node (center-most or highest degree)
2. Compute BFS distances from seed
3. Keep nodes at even distances (0, 2, 4, ...)
4. Discard nodes at odd distances (1, 3, 5, ...)

This creates a checkerboard-like pattern that preserves connectivity.

## Key Features

✅ **Compatible with your existing setup**: Works with AeroDataset and your data format
✅ **No manual coarsening**: Automatically creates multi-scale hierarchy
✅ **Efficient**: Preprocessing done once, not during training
✅ **Skip connections**: U-Net style for better gradient flow
✅ **Flexible**: Configurable number of levels and dimensions

## Differences from Original BSMS-GNN

This implementation is adapted to work with your codebase:

1. **Model structure**: Uses your MLP and activation functions
2. **Data format**: Works with PyTorch Geometric Data objects
3. **Training loop**: Compatible with your existing training code
4. **Edge features**: Handles your edge attributes correctly

## Current Limitations

⚠️ **Batch size = 1**: Current implementation processes one graph at a time
- Batching multi-scale graphs is complex
- For production, implement proper batching

⚠️ **Edge pooling**: Simple approach (zeros for coarse edges)
- Could be improved with learned edge pooling
- Original BSMS recomputes edge features

## Performance Tips

1. **Cache preprocessing**: Set `cache=True` in BSMSDatasetWrapper
2. **Adjust num_levels**: Start with 2-3 levels, increase for larger meshes
3. **Hidden dimensions**: Use 128-256 for good capacity
4. **Learning rate**: Start with 1e-4, adjust as needed

## Integration with Your Training Script

To integrate into `train.py`:

```python
from models.bsms_mgn import BSMS_MeshGraphNet
from models.bsms_dataset_wrapper import prepare_bsms_data

# In your model creation section:
if model_name == 'bsms_mgn':
    # Wrap datasets
    train_set = prepare_bsms_data(train_set, num_levels=3)
    val_set = prepare_bsms_data(val_set, num_levels=3)
    test_set = prepare_bsms_data(test_set, num_levels=3)

    # Create model
    model = BSMS_MeshGraphNet(
        input_node_dim=input_node_dim,
        input_edge_dim=input_edge_dim,
        output_node_dim=output_node_dim,
        num_levels=model_config.get('num_levels', 3),
        latent_dim=model_config.get('hidden_dim', 128),
        hidden_dim=model_config.get('hidden_dim', 128),
        pos_dim=pos_dim
    )
```

## References

```bibtex
@inproceedings{cao2023bsms,
  title={Efficient Learning of Mesh-Based Physical Simulation with Bi-Stride Multi-Scale Graph Neural Network},
  author={Cao, Yadi and Chai, Menglei and Li, Minchen and Jiang, Chenfanfu},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2023}
}
```

## Troubleshooting

**Issue**: Out of memory
**Solution**: Reduce `num_levels`, `latent_dim`, or use gradient checkpointing

**Issue**: Poor performance
**Solution**: Check that multi-scale preprocessing is working correctly, verify graph connectivity at each level

**Issue**: Multi_data error
**Solution**: Ensure you're using BSMSDatasetWrapper to preprocess data

## Questions?

Check the example script: `example_bsms_train.py`
Original BSMS-GNN: https://github.com/Eydcao/BSMS-GNN
