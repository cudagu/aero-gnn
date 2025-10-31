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

## Quick Start

### Training with BSMS

BSMS is fully integrated into the main training pipeline. Simply use an experiment configured with `model: bsms_mgn`:

```bash
# Train using the BSMS model
python train.py --exp airfoil_bsms_mgn
```

The training script automatically:
1. Wraps datasets with multi-scale preprocessing
2. Uses BSMS-compatible data loaders
3. Handles the multi_data dict during training
4. Saves model weights and normalization stats

### Configuration

Update your experiment in `config.yaml`:

```yaml
experiments:
  airfoil_bsms_mgn:
    dataset: airfoil_2d
    model: bsms_mgn
    training: default
    mach: [0.86]
    alpha: [3]
    data_dir: /path/to/data
    batch_size: 1  # BSMS requires batch_size=1
    epochs: 1000
    random_seed: 402
    test_split: 0.2
    num_levels: 3  # Number of coarsening levels
```

### Manual Usage (Advanced)

If you need to use BSMS outside the main pipeline:

```python
from dataset import AeroDataset
from models.bsms_mgn import BSMS_MeshGraphNet
from models.bsms_dataset_wrapper import prepare_bsms_data, BSMSDataLoader

# Create and wrap dataset
dataset = AeroDataset(data_dir='path/to/data', dataset_type='airfoil_2d', params=config)
bsms_dataset = prepare_bsms_data(dataset, num_levels=3)
loader = BSMSDataLoader(bsms_dataset, batch_size=1, shuffle=True)

# Create model
sample = next(iter(loader))
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
predictions = model(sample.x, sample.edge_attr, sample.edge_index, sample.multi_data)
```

## Architecture Details

### Edge Coarsening Strategy

BSMS uses the **A² method** from the original paper for edge coarsening:

1. **Square adjacency matrix**: Compute A² to capture 2-hop neighbors
2. **Pool edges**: Keep edges where both endpoints are selected nodes

This approach maintains mesh connectivity by connecting nodes that are 1-hop or 2-hop neighbors in the original graph, preventing the connectivity loss that would occur with naive edge filtering.

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

## Integration Status

✅ **BSMS is fully integrated into the main training pipeline!**

The following components automatically handle BSMS:
- **train.py**: Automatically wraps datasets and uses BSMS loaders when `model.name == 'bsms_mgn'`
- **utils.py**: `train()` and `evaluate()` functions handle multi_data dict
- **inference.py**: `predict_single()` handles BSMS model inference
- **utils.py**: `load_model_and_data()` wraps test sets for BSMS inference

No manual integration needed - just configure your experiment in config.yaml!

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

- Training with BSMS: Use `python train.py --exp airfoil_bsms_mgn`
- Original BSMS-GNN paper: https://github.com/Eydcao/BSMS-GNN
- See main CLAUDE.md for complete integration details
