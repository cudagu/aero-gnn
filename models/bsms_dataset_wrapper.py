"""
Dataset wrapper for BSMS model that preprocesses multi-scale graphs.

This wrapper should be used to preprocess your existing AeroDataset
for use with BSMS_MeshGraphNet.
"""

import torch
from torch_geometric.data import Dataset, Data
from models.bsms_mgn import MultiScaleGraphPreprocessor


class BSMSDatasetWrapper(Dataset):
    """
    Wraps an existing PyTorch Geometric dataset to add multi-scale preprocessing.

    This should be used to wrap your AeroDataset before training with BSMS model.
    """

    def __init__(self, base_dataset, num_levels=3, cache=True):
        """
        Args:
            base_dataset: Original dataset (e.g., AeroDataset)
            num_levels: Number of coarsening levels
            cache: Whether to cache preprocessed multi-scale graphs
        """
        super().__init__()
        self.base_dataset = base_dataset
        self.num_levels = num_levels
        self.cache = cache

        self.preprocessor = MultiScaleGraphPreprocessor(num_levels=num_levels)

        # Cache for preprocessed graphs
        self._cache = {} if cache else None

        print(f"BSMSDatasetWrapper initialized with {num_levels} levels")
        print(f"Base dataset size: {len(base_dataset)}")

    def len(self):
        return len(self.base_dataset)

    def get(self, idx):
        """
        Get preprocessed data with multi-scale graph hierarchy.

        Returns a Data object with additional attributes:
            - multi_data: Dictionary with multi-scale graph structure
        """
        # Check cache first
        if self.cache and idx in self._cache:
            return self._cache[idx]

        # Get base data
        data = self.base_dataset[idx]

        # Preprocess to create multi-scale hierarchy
        multi_data = self.preprocessor.create_multiscale_graph(data)

        # Create new data object with multi_data attribute
        processed_data = Data(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            pos=data.pos,
            y=data.y,
            multi_data=multi_data
        )

        # Copy additional attributes
        for key in data.keys:
            if key not in ['x', 'edge_index', 'edge_attr', 'pos', 'y']:
                setattr(processed_data, key, getattr(data, key))

        # Cache if enabled
        if self.cache:
            self._cache[idx] = processed_data

        return processed_data


def collate_bsms_batch(batch_list):
    """
    Custom collate function for batching BSMS data.

    Note: For simplicity, this returns a list of data objects.
    For true batching, you would need to batch multi_data structures.
    """
    # For now, return list (process one at a time)
    # True batching of multi-scale graphs is more complex
    return batch_list


class BSMSDataLoader:
    """
    Simple data loader for BSMS that processes graphs one at a time.

    For production use, implement proper batching of multi-scale graphs.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        """
        Args:
            dataset: BSMSDatasetWrapper instance
            batch_size: Batch size (currently only supports 1)
            shuffle: Whether to shuffle data
        """
        if batch_size != 1:
            print("Warning: BSMS currently only supports batch_size=1")
            print("Setting batch_size=1")
            batch_size = 1

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indices = list(range(len(dataset)))

    def __iter__(self):
        if self.shuffle:
            import random
            random.shuffle(self.indices)

        for idx in self.indices:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def prepare_bsms_data(base_dataset, num_levels=3):
    """
    Helper function to prepare dataset for BSMS model.

    Args:
        base_dataset: Original AeroDataset
        num_levels: Number of coarsening levels

    Returns:
        bsms_dataset: Wrapped dataset with multi-scale preprocessing
    """
    return BSMSDatasetWrapper(base_dataset, num_levels=num_levels, cache=True)


# Example usage:
if __name__ == "__main__":
    """
    Example of how to use BSMS dataset wrapper with your existing setup.
    """
    import sys
    sys.path.append('..')

    from dataset import AeroDataset
    import yaml

    # Load your configuration
    with open('../configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create base dataset
    base_dataset = AeroDataset(
        data_dir=config['dataset']['data_dir'],
        dataset_type='airfoil_2d',
        params=config
    )

    # Wrap with BSMS preprocessing
    bsms_dataset = prepare_bsms_data(base_dataset, num_levels=3)

    # Test getting one sample
    sample = bsms_dataset[0]

    print("\nSample data structure:")
    print(f"  Node features: {sample.x.shape}")
    print(f"  Edge index: {sample.edge_index.shape}")
    print(f"  Edge features: {sample.edge_attr.shape}")
    print(f"  Positions: {sample.pos.shape}")
    print(f"  Targets: {sample.y.shape}")

    print("\nMulti-scale structure:")
    multi_data = sample.multi_data
    print(f"  Number of levels: {len(multi_data['num_nodes'])}")
    for i, n in enumerate(multi_data['num_nodes']):
        print(f"    Level {i}: {n} nodes, {multi_data['edge_indices'][i].shape[1]} edges")
