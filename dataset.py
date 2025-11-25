import torch
import torch_geometric
import glob
import os
from pathlib import Path
from typing import Optional, List, Tuple
import random
from collections import defaultdict
import pyvista as pv
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import Dataset
from utils import plot_adjacency_matrix
import networkx as nx
import numpy as np

from tqdm import tqdm


class AeroDataset(Dataset):
    def __init__(self, data_dir: str, dataset_type: str, params = None, dtype: torch.dtype = torch.float32):
        """Initialize AeroDataset.
        
        Args:
            data_dir: Directory containing the dataset
            dataset_type: Type of dataset ('airfoil_2d', 'missile_3d', 'ahmed_body')
            params: Configuration parameters
            dtype: Data type for tensors (torch.float32 or torch.float64)
        """
        super().__init__()
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.params = params or {}
        self.data_list = []
        self.dtype = dtype  # Store data type for tensor creation
        
        if dataset_type == 'airfoil_2d':
            self._load_airfoil_2d()
        elif dataset_type == 'missile_3d1':
            self._load_missile_3d()
        elif dataset_type == 'ahmed_body':
            self._load_ahmed_body()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        plot_adjacency_matrix(self.data_list[0], title=f"{dataset_type} Sample Graph", save_path=f"{dataset_type}_sample_graph")

        if params["training"].get("reordering") == "rcm":
            print("Reordering graphs using RCM...")
            self.reorder_graphs(self.data_list)

        # NOTE: Clustering coefficient calculation is expensive - only use for analysis
        # clustering_coeffs = [calculate_clustering_coefficient(data) for data in self.data_list]
        # print(f"Clustering coefficients: {clustering_coeffs}")

        # plot_adjacency_matrix(self.data_list[0], title=f"{dataset_type} Sample Graph After Reordering", save_path=f"{dataset_type}_sample_graph_reordered")

    def convert_edge_index_to_bsr(self, edge_index: torch.Tensor, num_nodes: int,
                                   edge_attr: Optional[torch.Tensor] = None,
                                   block_size: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Convert edge_index to Block Compressed Sparse Row (BSR) format.

        BSR format stores sparse matrices in blocks for better memory access patterns.
        For block_size=1, this is equivalent to standard CSR format.

        Args:
            edge_index: [2, num_edges] tensor with source and target indices
            num_nodes: Total number of nodes in the graph
            edge_attr: [num_edges, edge_dim] optional edge attributes to reorder
            block_size: Block size for BSR format (default=1 for CSR)

        Returns:
            bsr_crow: [num_node_blocks + 1] row pointer array indicating where each row's blocks start
            bsr_col: [num_blocks] column indices for each block
            sorted_edge_indices: [num_edges] indices to reorder original edges to BSR order
            reordered_edge_attr: [num_edges, edge_dim] edge attributes reordered to match BSR structure (if edge_attr provided)
        """
        device = edge_index.device
        dtype_idx = edge_index.dtype

        # Ensure edge_index is [2, num_edges] format
        if edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges], got {edge_index.shape}")

        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.shape[1]

        # Sort edges by (source, destination) for CSR/BSR ordering
        # This groups all edges from the same source node together
        sorted_indices = torch.argsort(src * num_nodes + dst)
        src_sorted = src[sorted_indices]
        dst_sorted = dst[sorted_indices]

        # Compute row pointers (crow indices)
        # crow[i] indicates the starting position of edges from node i
        # crow has length num_nodes + 1, where crow[-1] = num_edges
        bsr_crow = torch.zeros(num_nodes + 1, dtype=dtype_idx, device=device)

        # Count edges per source node using bincount
        # This is equivalent to computing the histogram of source nodes
        edge_counts = torch.bincount(src_sorted, minlength=num_nodes)

        # Cumulative sum to get crow indices
        bsr_crow[1:] = torch.cumsum(edge_counts, dim=0)

        # Column indices are just the sorted destination nodes
        bsr_col = dst_sorted

        # Reorder edge attributes if provided
        reordered_edge_attr = None
        if edge_attr is not None:
            reordered_edge_attr = edge_attr[sorted_indices]

        return bsr_crow, bsr_col, sorted_indices, reordered_edge_attr

    def compute_edge_attr(self, data):
        """Compute edge attributes based on node positions.
        
        Args:
            data: PyTorch Geometric Data object with pos and edge_index
            
        Returns:
            edge_attr: Edge attributes tensor of shape [num_edges, edge_dim]
        """
        edge_index = data.edge_index
        pos = data.pos
        
        # Get source and target node positions
        source_pos = pos[edge_index[0]]  # [num_edges, pos_dim]
        target_pos = pos[edge_index[1]]  # [num_edges, pos_dim]
        
        # Compute edge vector (relative position)
        edge_vec = target_pos - source_pos  # [num_edges, pos_dim]
        
        # Compute edge length
        edge_length = torch.norm(edge_vec, dim=1, keepdim=True)  # [num_edges, 1]
        
        # Concatenate edge features: [edge_vec, edge_length]
        edge_attr = torch.cat([edge_vec, edge_length], dim=1)
        
        return edge_attr

    def compute_node_features(self, data):
        """Compute node features based on pos, normals, and other attributes.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            x: Node features tensor of shape [num_nodes, node_dim]
        """
        features = []
        
        # Add position coordinates
        features.append(data.pos)  # [num_nodes, 2] for 2D or [num_nodes, 3] for 3D
        
        # Add normals if available
        if hasattr(data, 'normals'):
            features.append(data.normals)
            
        # Add global parameters if available (broadcast to all nodes)
        var_keys = self.params["dataset"].get("var_keys", [])
        if var_keys:
            for key in var_keys:
                if hasattr(data, key):
                    value = getattr(data, key)
                    # Convert to tensor and ensure it's a scalar
                    if not isinstance(value, torch.Tensor):
                        feat = torch.tensor([value], dtype=self.dtype)
                    else:
                        feat = value if value.dim() > 0 else value.unsqueeze(0)
                        # Convert to correct dtype if needed
                        if feat.dtype != self.dtype:
                            feat = feat.to(self.dtype)
                    
                    # Broadcast to all nodes: [num_nodes, 1]
                    feat = feat.unsqueeze(0).expand(data.pos.size(0), -1)
                    features.append(feat)
        
        # Concatenate all features
        x = torch.cat(features, dim=1)
        
        return x
    
    def _load_airfoil_2d(self):
        files = glob.glob(os.path.join(self.data_dir, "*/*/walls_Surf64.vtu"))
        print(f"Found {len(files)} airfoil 2D files")
        
        # Get parameter ranges
        mach_range = self.params["dataset"].get('mach', None)
        alpha_range = self.params["dataset"].get('alpha', None)
        
        filtered_files = []
        for file in files:
            mach, alpha = Path(file).parts[-2].split('_')[-2:]
            mach = float(mach)
            alpha = float(alpha)
            
            # Filter based on ranges
            if mach_range is not None:
                if len(mach_range) == 1:  # Single value
                    if abs(mach - mach_range[0]) > 1e-6:
                        continue
                elif len(mach_range) == 2:  # Range [min, max]
                    if mach < mach_range[0] or mach > mach_range[1]:
                        continue
            
            if alpha_range is not None:
                if len(alpha_range) == 1:  # Single value
                    if abs(alpha - alpha_range[0]) > 1e-6:
                        continue
                elif len(alpha_range) == 2:  # Range [min, max]
                    if alpha < alpha_range[0] or alpha > alpha_range[1]:
                        continue
            
            filtered_files.append(file)
        
        print(f"Filtered to {len(filtered_files)} files based on parameters")
        if mach_range:
            print(f"  Mach range: {mach_range}")
        if alpha_range:
            print(f"  Alpha range: {alpha_range}")
        
        pbar = tqdm(filtered_files, desc="Loading Airfoil 2D files")
        for file in pbar:
            mach, alpha = Path(file).parts[-2].split('_')[-2:]
            mach = float(mach)
            alpha = float(alpha)
            airfoil_name = Path(file).parts[-3]
            pbar.set_description(f"Loading {airfoil_name} M={mach:.2f} α={alpha:.1f}")
            data = read_2d_mesh(file, airfoil_name=airfoil_name, dtype=self.dtype)
            data.mach = mach
            data.alpha = alpha
            
            # Compute node features and edge attributes
            data.x = self.compute_node_features(data)
            data.edge_attr = self.compute_edge_attr(data)
            
            self.data_list.append(data)
    
    def _load_missile_3d(self):
        files = glob.glob(os.path.join(self.data_dir, "**/*.vtp"), recursive=True)
        print(f"Found {len(files)} missile 3D files")
        
        # Get parameter ranges
        mach_range = self.params.get('mach', None)
        alpha_range = self.params.get('alpha', None)
        beta_range = self.params.get('beta', None)
        
        filtered_files = []
        for file in files:
            try:
                path_parts = Path(file).parts
                filename = path_parts[-2]
                
                # Extract mach, alpha, beta from filename with structure "m0.85_b0_a0"
                mach = float([s for s in filename.split('_') if s.startswith('m')][0][1:])
                beta = float([s for s in filename.split('_') if s.startswith('b')][0][1:])
                alpha = float([s for s in filename.split('_') if s.startswith('a')][0][1:])

                # Filter based on ranges
                if mach_range is not None:
                    if len(mach_range) == 1:  # Single value
                        if abs(mach - mach_range[0]) > 1e-6:
                            continue
                    elif len(mach_range) == 2:  # Range [min, max]
                        if mach < mach_range[0] or mach > mach_range[1]:
                            continue
                
                if alpha_range is not None:
                    if len(alpha_range) == 1:  # Single value
                        if abs(alpha - alpha_range[0]) > 1e-6:
                            continue
                    elif len(alpha_range) == 2:  # Range [min, max]
                        if alpha < alpha_range[0] or alpha > alpha_range[1]:
                            continue
                
                if beta_range is not None:
                    if len(beta_range) == 1:  # Single value
                        if abs(beta - beta_range[0]) > 1e-6:
                            continue
                    elif len(beta_range) == 2:  # Range [min, max]
                        if beta < beta_range[0] or beta > beta_range[1]:
                            continue
                
                filtered_files.append(file)
                
            except (ValueError, IndexError):
                # If can't extract parameters, skip file or include all
                filtered_files.append(file)
                continue
        
        print(f"Filtered to {len(filtered_files)} files based on parameters")
        if mach_range:
            print(f"  Mach range: {mach_range}")
        if alpha_range:
            print(f"  Alpha range: {alpha_range}")
        if beta_range:
            print(f"  Beta range: {beta_range}")
            
        train_num_samples = self.params["training"].get("train_num_samples")
        val_num_samples = self.params["training"].get("val_num_samples")
        test_num_samples = self.params["training"].get("test_num_samples")
        
        train_num_samples = None if train_num_samples == "None" else train_num_samples
        val_num_samples = None if val_num_samples == "None" else val_num_samples
        test_num_samples = None if test_num_samples == "None" else test_num_samples
        

        if train_num_samples is not None or val_num_samples is not None or test_num_samples is not None:
            sampled_files = []
            split_counter = 0
            split_limit = int(train_num_samples) + int(val_num_samples) + int(test_num_samples)

            for file_path in filtered_files:
                if split_counter < split_limit:
                    sampled_files.append(file_path)
                    split_counter += 1
            filtered_files = sampled_files
        pbar = tqdm(filtered_files, desc="Loading Missile 3D files")
        for file in pbar:
            filename = Path(file).name
            pbar.set_description(f"Loading {filename}")
            data = read_3d_mesh(file, dtype=self.dtype)
            
            # Compute node features and edge attributes
            data.x = self.compute_node_features(data)
            data.edge_attr = self.compute_edge_attr(data)
            
            self.data_list.append(data)
    
    def _load_ahmed_body(self):
        """Load ahmed_body dataset from pre-split train/validation/test directories.
        """
                    
        splits_to_load = ['train', 'validation', 'test']
        
        all_files = []
        for split_name in splits_to_load:
            split_dir = os.path.join(self.data_dir, split_name)
            split_dir_info = os.path.join(self.data_dir, split_dir+'_info')
            if not os.path.exists(split_dir):
                print(f"Warning: Split directory not found: {split_dir}")
                continue

            # Find all .vtp files in this split and also their corresponding info files in split_dir_info
            files = glob.glob(os.path.join(split_dir, "*.vtp"))
            print(f"Found {len(files)} files in {split_name} split")
            all_files.extend([(f, split_name) for f in files])
        
        print(f"Total Ahmed Body files to load: {len(all_files)}")
        #sample all_files based on params train/val/test_num_samples if specified
        train_num_samples = self.params["training"].get("train_num_samples")
        val_num_samples = self.params["training"].get("val_num_samples")
        test_num_samples = self.params["training"].get("test_num_samples")
        
        
        
        if train_num_samples is not None or val_num_samples is not None or test_num_samples is not None:
            sampled_files = []
            split_counters = {'train': 0, 'validation': 0, 'test': 0}
            split_limits = {
                'train': train_num_samples,
                'validation': val_num_samples,
                'test': test_num_samples
            }
            
            for file_path, split_name in all_files:
                limit = split_limits.get(split_name)
                if limit is None or split_counters[split_name] < limit:
                    sampled_files.append((file_path, split_name))
                    split_counters[split_name] += 1
            all_files = sampled_files
            print(f"After sampling based on specified numbers:")
            print(f"  Train samples: {split_counters['train']}")
            print(f"  Validation samples: {split_counters['validation']}")
            print(f"  Test samples: {split_counters['test']}")
        
        pbar = tqdm(all_files, desc="Loading Ahmed Body files")
        for file_path, split_name in pbar:
                   
            case_no = Path(file_path).name.split('.')[0]  # Get case number without extension
            pbar.set_description(f"Loading {split_name}/{case_no}")
            
            try:
                
                info_file_path = os.path.join(self.data_dir, split_name+'_info', case_no+'_info.txt')
                with open(info_file_path, 'r') as f:
                    info_lines = f.readlines()
                    #Info file has lines like: Length : 4.267 Width : 339.0 Height : 283.0 Velocity : 30.0 etc
                    info_dict = {}
                    for line in info_lines:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            info_dict[key.strip()] = float(value.strip())
                
                data = read_AhmedBody(file_path, dtype=self.dtype)
                
                # Add split information as metadata
                data.split = split_name
                data.case_no = case_no
                
                
                # Add relevant info as attributes
                for key, value in info_dict.items():
                    setattr(data, key, value)


                # Compute node features and edge attributes
                data.x = self.compute_node_features(data)
                data.edge_attr = self.compute_edge_attr(data)
                
                self.data_list.append(data)
            except (KeyError, ValueError) as e:
                print(f"\nWarning: Failed to load {file_path}: {e}")
                continue

    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    def get_airfoil_names(self):
        """Get list of unique airfoil names in the dataset (for airfoil_2d only)."""
        if self.dataset_type != 'airfoil_2d':
            raise ValueError("Airfoil names are only available for airfoil_2d dataset")
        
        airfoil_names = set()
        for data in self.data_list:
            if hasattr(data, 'airfoil'):
                airfoil_names.add(data.airfoil)
        
        return list(airfoil_names)
    
    def compute_normalization_stats(self, data_list):
        """Compute normalization statistics from a list of data objects.
        
        Args:
            data_list: List of PyTorch Geometric Data objects
            
        Returns:
            stats: Dictionary containing mean and std for node features and edge attributes
        """
        # Collect all node features and edge attributes
        x_stack = torch.vstack([data.x for data in data_list])  # [total_nodes, node_dim]
        edge_attr_stack = torch.vstack([data.edge_attr for data in data_list])  # [total_edges, edge_dim]
        y_stack = torch.vstack([data.y for data in data_list])  # [total_nodes, target_dim]
        
        x_std, x_mean = torch.std_mean(x_stack, dim=0)  # [node_dim]
        edge_attr_std, edge_attr_mean = torch.std_mean(edge_attr_stack, dim=0)  # [edge_dim]
        y_std, y_mean = torch.std_mean(y_stack, dim=0)  # [target_dim]     
        
        # Compute statistics
        stats = {
            'node_mean': x_mean,
            'node_std': x_std,
            'edge_mean': edge_attr_mean,
            'edge_std': edge_attr_std,
            'target_mean': y_mean,
            'target_std': y_std,
        }
        
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        stats['node_std'] = torch.clamp(stats['node_std'], min=eps)
        stats['edge_std'] = torch.clamp(stats['edge_std'], min=eps)
        stats['target_std'] = torch.clamp(stats['target_std'], min=eps)
        
        return stats
    
    def normalize_data(self, data_list, stats):
        """Normalize data using provided statistics.
        
        Args:
            data_list: List of data objects to normalize
            stats: Normalization statistics dictionary
        """
        for data in data_list:
            # Normalize node features
            data.x = (data.x - stats['node_mean']) / stats['node_std']
            
            # Normalize edge attributes
            data.edge_attr = (data.edge_attr - stats['edge_mean']) / stats['edge_std']
            
            # Normalize targets
            data.y = (data.y - stats['target_mean']) / stats['target_std']
    
    def denormalize_predictions(self, predictions, stats):
        """Denormalize predictions back to original scale.
        
        Args:
            predictions: Normalized predictions tensor
            stats: Normalization statistics dictionary
            
        Returns:
            Denormalized predictions
        """
        return predictions * stats['target_std'] + stats['target_mean']
    
    def split_airfoil_by_name(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if self.dataset_type != 'airfoil_2d':
            raise ValueError("This split method is only for airfoil_2d dataset")
        
        # Group data by airfoil name
        airfoil_groups = defaultdict(list)
        for data in self.data_list:
            airfoil_name = data.airfoil
            airfoil_groups[airfoil_name].append(data)
        
        airfoil_names = list(airfoil_groups.keys())
        random.seed(random_seed)
        random.shuffle(airfoil_names)
        
        n_airfoils = len(airfoil_names)
        n_train = int(n_airfoils * train_ratio)
        n_val = int(n_airfoils * val_ratio)
        
        train_airfoils = airfoil_names[:n_train]
        val_airfoils = airfoil_names[n_train:n_train + n_val]
        test_airfoils = airfoil_names[n_train + n_val:]
        
        print(f"Dataset split by airfoil names:")

        
        # Create data lists for each split
        train_data = []
        val_data = []
        test_data = []
        
        for airfoil_name in train_airfoils:
            train_data.extend(airfoil_groups[airfoil_name])
        for airfoil_name in val_airfoils:
            val_data.extend(airfoil_groups[airfoil_name])
        for airfoil_name in test_airfoils:
            test_data.extend(airfoil_groups[airfoil_name])
        
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        print(f"  Test samples: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def split_generic(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
        """Generic split for non-airfoil datasets."""
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        data_list = self.data_list.copy()
        random.seed(random_seed)
        random.shuffle(data_list)
        
        n_total = len(data_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_data = data_list[:n_train]
        val_data = data_list[n_train:n_train + n_val]
        test_data = data_list[n_train + n_val:]
        
        print(f"Generic dataset split:")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        print(f"  Test samples: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def split_presplit(self):
        """Split dataset that is already pre-split (like ahmed_body).
        
        Returns data split by the 'split' attribute that was set during loading.
        """
        train_data = []
        val_data = []
        test_data = []
        
        for data in self.data_list:
            if hasattr(data, 'split'):
                if data.split == 'train':
                    train_data.append(data)
                elif data.split == 'validation':
                    val_data.append(data)
                elif data.split == 'test':
                    test_data.append(data)
            else:
                # If no split attribute, default to train
                train_data.append(data)
        
        print(f"Pre-split dataset:")
        print(f"  Train samples: {len(train_data)}")
        print(f"  Val samples: {len(val_data)}")
        print(f"  Test samples: {len(test_data)}")
        
        return train_data, val_data, test_data
    
    def reorder_graphs(self, data_list: List[torch_geometric.data.Data]):
        """Reorder graphs using RCM algorithm for better cache locality.

        The Reverse Cuthill-McKee (RCM) algorithm reorders nodes to minimize
        the bandwidth of the adjacency matrix, improving spatial locality for
        better cache performance during graph traversal.

        Args:
            data_list: List of PyTorch Geometric Data objects to reorder in-place
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import reverse_cuthill_mckee

        pbar = tqdm(data_list, desc="Applying RCM reordering")
        for i, data in enumerate(pbar):
            num_nodes = data.num_nodes
            edge_index = data.edge_index.cpu().numpy()

            # Create scipy sparse adjacency matrix (CSR format for efficiency)
            adj = csr_matrix(
                (
                    torch.ones(edge_index.shape[1]).numpy(),  # values
                    (edge_index[0], edge_index[1])            # (row, col) indices
                ),
                shape=(num_nodes, num_nodes)
            )

            # Compute RCM ordering
            perm = reverse_cuthill_mckee(adj, symmetric_mode=True)

            # Convert permutation to torch tensor (copy to ensure positive strides)
            perm_tensor = torch.from_numpy(perm.copy()).long()

            # Reorder node features
            data.x = data.x[perm_tensor]
            data.pos = data.pos[perm_tensor]
            if hasattr(data, 'normals'):
                data.normals = data.normals[perm_tensor]
            if hasattr(data, 'y'):
                data.y = data.y[perm_tensor]

            # Create inverse mapping (old index -> new index)
            old_to_new = torch.empty(num_nodes, dtype=torch.long)
            old_to_new[perm_tensor] = torch.arange(num_nodes, dtype=torch.long)

            # Reorder edge_index using vectorized operations
            data.edge_index = old_to_new[data.edge_index]

            # Recompute edge attributes after reordering
            # Edge attributes depend on node positions, so they must be recomputed
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = self.compute_edge_attr(data)

            data_list[i] = data
        

def read_2d_mesh(file_path: str, airfoil_name: str, dtype: torch.dtype = torch.float32) -> torch_geometric.data.Data:
    """Read an airfoil case and return a 2D surface graph.

    Assumes the file contains fields: 'tau' (shear stress), 'P' (pressure), 'T' (temperature).
    
    Args:
        file_path: Path to the VTU file
        airfoil_name: Name of the airfoil
        dtype: Data type for tensors (torch.float32 or torch.float64)
    """
    mesh = pv.read(file_path)
    surface = mesh.extract_surface()
    surface = surface.compute_normals(
        cell_normals=False, point_normals=True, inplace=True, flip_normals=True
    )
    # Slice close to the z=0 plane, useful if the dataset is an extruded 3D surface.
    slc = surface.slice(normal=(0.0, 0.0, 1.0))
    slc = slc.cell_data_to_point_data()

    edges = slc.extract_all_edges(use_all_points=True, clear_data=True)
    edge_index = torch.tensor(
        edges.lines.reshape(-1, 3)[:, 1:].T, dtype=torch.long
    )
    pos_np = slc.points[:, :2]
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index, num_nodes=pos_np.shape[0])

    pos = torch.tensor(pos_np, dtype=dtype)
    normals = torch.tensor(slc.point_normals[:, :2], dtype=dtype)

    shear_stress = torch.tensor(slc["tau"][:, :2], dtype=dtype)
    pressure = torch.tensor(slc["P"][:, None], dtype=dtype)
    temperature = torch.tensor(slc["t"][:, None], dtype=dtype)

    data = torch_geometric.data.Data(
        edge_index=edge_index,
        pos=pos,
        normals=normals,
        airfoil = airfoil_name,
        y=torch.cat([pressure, shear_stress, temperature], dim=-1),
        
    )
    return data

def read_3d_mesh(file_path: str, dtype: torch.dtype = torch.float32) -> torch_geometric.data.Data:
    """Read a 3D surface graph (e.g., missile) from a VTP/VTU/STL with fields.

    Returns Data with pos (N,3), normals (N,3), edge_index, and y = [P, tau_x, tau_y, tau_z, T].
    
    Args:
        file_path: Path to the mesh file
        dtype: Data type for tensors (torch.float32 or torch.float64)
    """
    mesh = pv.read(file_path)
    surface = mesh.extract_surface()
    surface = surface.compute_normals(
        cell_normals=False, point_normals=True, inplace=True, flip_normals=True
    )
    surface = surface.cell_data_to_point_data()

    edges = surface.extract_all_edges(use_all_points=True, clear_data=True)
    edge_index = torch.tensor(
        edges.lines.reshape(-1, 3)[:, 1:].T, dtype=torch.long
    )
    pos_np = surface.points
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index, num_nodes=pos_np.shape[0])

    pos = torch.tensor(pos_np, dtype=dtype)
    normals = torch.tensor(surface.point_normals, dtype=dtype)
    shear_stress = torch.tensor(surface["tau"], dtype=dtype)
    pressure = torch.tensor(surface["P"][:, None], dtype=dtype)
    temperature = torch.tensor(surface["t"][:, None], dtype=dtype)

    data = torch_geometric.data.Data(
        edge_index=edge_index,
        pos=pos,
        normals=normals,
        y=torch.cat([pressure, shear_stress, temperature], dim=-1),
    )
    return data

def read_AhmedBody(file_path: str, dtype: torch.dtype = torch.float32) -> torch_geometric.data.Data:
    """Read an AhmedBody case with 'wallShearStress', 'P'.
    
    Args:
        file_path: Path to the mesh file
        dtype: Data type for tensors (torch.float32 or torch.float64)
    """
    mesh = pv.read(file_path)
    surface = mesh.extract_surface()
    surface = surface.compute_normals(
        cell_normals=False, point_normals=True, inplace=True, flip_normals=True
    )
    surface = surface.cell_data_to_point_data()

    edges = surface.extract_all_edges(use_all_points=True, clear_data=True)
    edge_index = torch.tensor(
        edges.lines.reshape(-1, 3)[:, 1:].T, dtype=torch.long
    )
    pos_np = surface.points
    if not is_undirected(edge_index):
        edge_index = to_undirected(edge_index, num_nodes=pos_np.shape[0])

    pos = torch.tensor(pos_np, dtype=dtype)
    normals = torch.tensor(surface.point_normals, dtype=dtype)
    shear_stress = torch.tensor(surface["wallShearStress"], dtype=dtype)
    pressure = torch.tensor(surface["p"][:, None], dtype=dtype)

    data = torch_geometric.data.Data(
        edge_index=edge_index,
        pos=pos,
        normals=normals,
        y=torch.cat([pressure, shear_stress], dim=-1),
    )
    return data

def create_datasets(data_dir: str, dataset_type: str, params: dict, dtype: torch.dtype = torch.float32):
    """Create train, val, test datasets with normalization.
    
    Args:
        data_dir: Directory containing the dataset
        dataset_type: Type of dataset
        params: Configuration parameters
        dtype: Data type for tensors (torch.float32 or torch.float64)
    """
    full_dataset = AeroDataset(data_dir, dataset_type, params, dtype=dtype)
    
    val_ratio = params['training'].get('validation_split')
    test_ratio = params['training'].get('test_split')
    train_ratio = 1.0 - val_ratio - test_ratio
    random_seed = params['training'].get('random_seed')
    
    if dataset_type == 'airfoil_2d':
        train_data, val_data, test_data = full_dataset.split_airfoil_by_name(
            train_ratio, val_ratio, test_ratio, random_seed
        )
    elif dataset_type == 'ahmed_body':
        # Ahmed body data is already pre-split into train/validation/test directories
        train_data, val_data, test_data = full_dataset.split_presplit()
    elif dataset_type == 'missile_3d1':
        train_data, val_data, test_data = full_dataset.split_generic(
            train_ratio, val_ratio, test_ratio, random_seed
        )
    else:
        train_data, val_data, test_data = full_dataset.split_generic(
            train_ratio, val_ratio, test_ratio, random_seed
        )
    
    # Compute normalization statistics from training data
    print("Computing normalization statistics from training data...")
    norm_stats = full_dataset.compute_normalization_stats(train_data)
    
    # Apply normalization to all datasets
    print("Applying normalization...")
    full_dataset.normalize_data(train_data, norm_stats)
    full_dataset.normalize_data(val_data, norm_stats)
    full_dataset.normalize_data(test_data, norm_stats)
    
    # Return data lists directly - PyTorch Geometric DataLoader works with lists of Data objects
    return train_data, val_data, test_data, norm_stats


def save_normalization_stats(norm_stats, file_path):
    """Save normalization statistics to file."""
    torch.save(norm_stats, file_path)
    print(f"Normalization statistics saved to {file_path}")


def load_normalization_stats(file_path):
    """Load normalization statistics from file."""
    norm_stats = torch.load(file_path)
    print(f"Normalization statistics loaded from {file_path}")
    return norm_stats


def calculate_clustering_coefficient(data: torch_geometric.data.Data) -> float:
    """
    Calculate the average clustering coefficient of a graph using NetworkX.

    The clustering coefficient measures how close the node's neighbors are to
    being a complete graph (clique).

    Args:
        data: PyTorch Geometric Data object with edge_index attribute

    Returns:
        Average clustering coefficient across all nodes
    """
    edge_index = data.edge_index.cpu().numpy()
    num_nodes = data.num_nodes

    # Create NetworkX undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(zip(edge_index[0], edge_index[1]))

    # Calculate average clustering coefficient
    return nx.average_clustering(G)