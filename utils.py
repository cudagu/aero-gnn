import os
import random
import torch
from torch import nn
import torch_geometric
from tqdm import tqdm
import copy
import pyvista as pv
import glob
from pathlib import Path
from torch_geometric.utils import is_undirected, to_undirected
from typing import List, Optional


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
    temperature = torch.tensor(surface["T"][:, None], dtype=dtype)

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

def get_experiment_config(params, configs):
    """Start from defaults and override with values declared in an experiment."""
    params = copy.deepcopy(params)

    dataset_name = params.pop("dataset")
    model_name = params.pop("model")
    training_name = params.pop("training")

    dataset_conf = copy.deepcopy(configs["dataset"][dataset_name])
    model_conf = copy.deepcopy(configs["model"][model_name])
    training_conf = copy.deepcopy(configs["training"][training_name])

    def take_overrides(base_conf):
        overrides = {}
        for key in list(params.keys()):
            if key in base_conf:
                overrides[key] = params.pop(key)
        return overrides

    dataset_conf.update(take_overrides(dataset_conf))
    dataset_conf["name"] = dataset_name

    model_conf.update(take_overrides(model_conf))
    model_conf["name"] = model_name

    training_conf.update(take_overrides(training_conf))
    training_conf["name"] = training_name

    result = {
        "dataset": dataset_conf,
        "model": model_conf,
        "training": training_conf,
    }

    if params:
        result["extras"] = params

    return result

def train(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        
        # Check if model needs batch parameter (for poolMGN and MeshGraphNet_v2)
        model_class = model.__class__.__name__
        if model_class in ['poolMGN', 'MeshGraphNet_v2']:
            pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)

        elif model_class in ['MLPNet']:
            pred = model(batch.x)
        else:
            pred = model(batch.x, batch.edge_attr, batch.edge_index)
            
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        
        # Check if model needs batch parameter (for poolMGN and MeshGraphNet_v2)
        model_class = model.__class__.__name__
        if model_class in ['poolMGN', 'MeshGraphNet_v2']:
            pred = model(batch.x, batch.edge_attr, batch.edge_index, batch.batch)
        elif model_class in ['MLPNet']:
            pred = model(batch.x)
        else:
            pred = model(batch.x, batch.edge_attr, batch.edge_index)
            
        loss = loss_fn(pred, batch.y)
        total_loss += loss.item()
    return total_loss / len(loader)


import matplotlib.pyplot as plt
import torch
import itertools
import json
import glob
from torch_geometric.loader import DataLoader

# Configure matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

def load_model_and_data(training_output_dir: str):
    """Load trained model and test data from training output directory."""
    
    # Load parameters
    params_path = os.path.join(training_output_dir, "experiment_params.json")
    with open(params_path, 'r') as f:
        params = json.load(f)
    
    # Load normalization stats
    norm_stats_path = os.path.join(training_output_dir, "normalization_stats.pt")
    norm_stats = torch.load(norm_stats_path, map_location='cpu')
    
    # Recreate test dataset
    from dataset import create_datasets
    _, _, test_set, _ = create_datasets(
        data_dir=params['dataset']['data_dir'],
        dataset_type=params['dataset']['name'], 
        params=params
    )
    
    # Load model architecture and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get dimensions from test data
    sample_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    sample_batch = next(iter(sample_loader))
    input_node_dim = sample_batch.x.shape[1]
    input_edge_dim = sample_batch.edge_attr.shape[1]
    output_node_dim = sample_batch.y.shape[1]
    
    # Recreate model
    model_name = params['model']['name']
    model_config = params['model']
    
    if model_name.lower() in ['mlp', 'mlpnet']:
        from models.baseline import MLPNet
        model = MLPNet(
            input_node_dim=input_node_dim,
            output_node_dim=output_node_dim,
            hidden_dim=model_config.get('hidden_dim'),
            num_layers=model_config.get('num_layers')
        )
    elif model_name.lower() in ['meshgraphnet', 'mgn']:
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
            dropout=model_config.get('dropout')
        )
    
    elif model_name.lower() in ['poolmgn']:
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

    else:
        raise ValueError(f"Unknown model type: {model_name}")
    
    # Load weights
    weights_path = os.path.join(training_output_dir, "model_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    
    return model, norm_stats, test_set, params, device


def find_latest_training_run(base_dir="training_runs"):
    """Find the most recent training run directory.
    Directory structure:
    base_dir/DD_MM_YY/training_folder
    
    """
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Training runs directory not found: {base_dir}")
    
    pattern = os.path.join(base_dir, "*", "*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    
    if not dirs:
        raise FileNotFoundError(f"No training runs found in {base_dir}")
    
    # Sort by modification time, most recent first
    dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return dirs[0]


def calculate_aero_coefficients_3d(
    surface,
    reference_area: float = 1.0,
    reference_length: float = 1.0,
    moment_center = None,
    dynamic_pressure: float = 1.0,
) -> dict:

    import numpy as np
    pressure_true     = surface.cell_data["p"]
    shear_stress_true = surface.cell_data["wallShearStress"]
    shear_stress_true = np.asarray(shear_stress_true)
    pressure_pred    = surface.cell_data["p_pred"]
    shear_stress_pred = surface.cell_data["wallShearStress_pred"]
    shear_stress_pred = np.asarray(shear_stress_pred)
    node_areas = surface.cell_data['Area']
    normals = surface.cell_data["Normals"]  

    

    pressure_force_true = pressure_true[:, np.newaxis] * normals * node_areas[:, np.newaxis]
    shear_force_true = -shear_stress_true * node_areas[:, np.newaxis]
    total_force_true = pressure_force_true + shear_force_true

    pressure_force_pred = pressure_pred[:, np.newaxis] * normals * node_areas[:, np.newaxis]
    shear_force_pred = -shear_stress_pred * node_areas[:, np.newaxis]
    total_force_pred = pressure_force_pred + shear_force_pred

    flow_dir = np.array([1, 0, 0])
    lift_dir = np.array([0, 1, 0])
    side_dir = np.array([0, 0, 1])

    F_axial_true  = np.dot(total_force_true, flow_dir)     # Axial (drag direction)
    F_normal_true = np.dot(total_force_true, lift_dir)    # Normal (lift direction)
    F_side_true   = np.dot(total_force_true, side_dir)      # Side force

    F_axial_pred  = np.dot(total_force_pred, flow_dir)     # Axial (drag direction)
    F_normal_pred = np.dot(total_force_pred, lift_dir)    # Normal (lift direction)
    F_side_pred   = np.dot(total_force_pred, side_dir)      # Side force
    
    F_axial_true = np.sum(F_axial_true, axis=0)
    F_normal_true = np.sum(F_normal_true, axis=0)
    F_side_true = np.sum(F_side_true, axis=0)

    F_axial_pred = np.sum(F_axial_pred, axis=0)
    F_normal_pred = np.sum(F_normal_pred, axis=0)
    F_side_pred = np.sum(F_side_pred, axis=0)

    CA_true = F_axial_true  / reference_area / dynamic_pressure
    CN_true = F_normal_true / reference_area / dynamic_pressure
    CY_true = F_side_true   / reference_area / dynamic_pressure

    CA_pred = F_axial_pred  / reference_area / dynamic_pressure
    CN_pred = F_normal_pred / reference_area / dynamic_pressure
    CY_pred = F_side_pred   / reference_area / dynamic_pressure
        
    return {
        'CA_true': float(CA_true),
        'CN_true': float(CN_true),
        'CY_true': float(CY_true),
        'CA_pred': float(CA_pred),
        'CN_pred': float(CN_pred),
        'CY_pred': float(CY_pred)
    }


def calculate_aero_coefficients_2d(
    test_data,
    pressure,
    shear_stress,
    reference_area: float = 1.0,
    reference_length: float = 1.0,
    moment_center = None,
    dynamic_pressure: float = 1.0
) -> dict:
    """
    Calculate aerodynamic force and moment coefficients from surface data.
    
    Args:
        test_data: 
        pressure: Pressure at each node (N, 1) or (N,)
        shear_stress: Shear stress at each node (N, 2) for 2D or (N, 3) for 3D
        reference_area: Reference area for normalization (default: 1.0)
        reference_length: Reference length for moment coefficients (default: 1.0)
        moment_center: Center for moment calculation (default: origin)
        dynamic_pressure: Dynamic pressure for normalization (default: 1.0)
    
    Returns:
        dict: Dictionary containing aerodynamic coefficients
            2D: {'CA', 'CN', 'Cm'}
            3D: {'CA', 'CN', 'CY', 'Cl', 'Cm', 'Cn'}
    """
    import numpy as np
    
    # Ensure tensors are on CPU and convert to numpy for easier manipulation
    pos = test_data.pos.cpu().numpy() if torch.is_tensor(test_data.pos) else test_data.pos
    pressure = pressure.cpu().numpy() if torch.is_tensor(pressure) else pressure
    shear_stress = shear_stress.cpu().numpy() if torch.is_tensor(shear_stress) else shear_stress
    normals = test_data.normals.cpu().numpy() if torch.is_tensor(test_data.normals) else test_data.normals
    edge_index = test_data.edge_index.cpu().numpy() if torch.is_tensor(test_data.edge_index) else test_data.edge_index
    
    # Flatten pressure if needed
    if pressure.ndim > 1:
        pressure = pressure.flatten()
    
    # Determine dimensionality
    ndim = pos.shape[1]
    is_2d = (ndim == 2)
    
    # Set moment center to origin if not provided
    if moment_center is None:
        moment_center = np.zeros(ndim)
    elif torch.is_tensor(moment_center):
        moment_center = moment_center.cpu().numpy()
    
   
    # For coefficient normalization, we use q_inf * S_ref
    # For simplicity, we'll use direct integration and normalize by reference values
    
    # Define body axis: x-axis along freestream direction
    # 2D: flow direction in x-y plane
    flow_dir = np.array([1, 0])
    normal_to_flow = np.array([0, 1])
    
    
    # Calculate panel areas using edge connectivity
    # For each edge, calculate the midpoint distance as representative "panel length/area"
    num_nodes = pos.shape[0]
    node_areas = np.zeros(num_nodes)

    
    for i in range(edge_index.shape[1]):
        n1, n2 = edge_index[0, i], edge_index[1, i]
        edge_length = np.linalg.norm(pos[n2] - pos[n1])
        # Distribute half the edge length to each node
        node_areas[n1] += edge_length / 2 * 1e-2
    
    # Calculate pressure forces: F_p = p * n * dA
    # Force per node
    pressure_force = pressure[:, np.newaxis] * normals * node_areas[:, np.newaxis]
    
    # Calculate shear forces: F_tau = -tau * dA
    # Shear stress already points in tangential direction
    shear_force = -shear_stress * node_areas[:, np.newaxis]
    
    # Total force per node
    total_force = pressure_force + shear_force
    
    # Integrate forces
    total_force_integrated = np.sum(total_force, axis=0)
    
    # Calculate moments about moment center
    # M = r × F, where r is position vector from moment center
    r = pos - moment_center
    

    # 2D: moment is scalar (out of plane, z-direction)
    # M_z = r_x * F_y - r_y * F_x
    moment_per_node = r[:, 0] * total_force[:, 1] - r[:, 1] * total_force[:, 0]
    total_moment = np.sum(moment_per_node)
    
    # Calculate axial and normal forces
    F_axial = np.dot(total_force_integrated, flow_dir)
    F_normal = np.dot(total_force_integrated, normal_to_flow)
    
    # Normalize by q_inf * S_ref
    CA = F_axial / reference_area / dynamic_pressure
    CN = F_normal / reference_area / dynamic_pressure 
    Cm = total_moment / (reference_area * reference_length) / dynamic_pressure
    
    return {
        'CA': float(CA),
        'CN': float(CN),
        'Cm': float(Cm)
    }


def plot_adjacency_matrix(
    data: torch_geometric.data.Data,
    title: str = "Graph Adjacency Matrix",
    figsize: tuple = (8, 8),
    save_path: str = None
) -> None:
    """
    Plot the adjacency matrix structure as a separate figure.

    Args:
        data: PyTorch Geometric Data object containing edge_index
        title: Title for the plot
        figsize: Figure size as (width, height) tuple
        save_path: Path to save the figure
    """
    import numpy as np
    from scipy.sparse import coo_matrix

    edge_index = data.edge_index.cpu().numpy()

    if hasattr(data, 'num_nodes') and data.num_nodes is not None:
        num_nodes = data.num_nodes
    else:
        num_nodes = int(edge_index.max()) + 1

    row, col = edge_index[0], edge_index[1]
    values = np.ones(len(row))
    adj_matrix = coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes))

    fig, ax = plt.subplots(figsize=figsize)

    max_display_nodes = 100000
    if num_nodes > max_display_nodes:
        sample_indices = np.random.choice(num_nodes, max_display_nodes, replace=False)
        sample_indices = np.sort(sample_indices)
        mask = np.isin(edge_index[0], sample_indices) & np.isin(edge_index[1], sample_indices)
        sampled_edges = edge_index[:, mask]
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sample_indices)}
        remapped_row = np.array([index_map[idx] for idx in sampled_edges[0]])
        remapped_col = np.array([index_map[idx] for idx in sampled_edges[1]])
        sampled_values = np.ones(len(remapped_row))
        sampled_adj = coo_matrix(
            (sampled_values, (remapped_row, remapped_col)),
            shape=(max_display_nodes, max_display_nodes)
        )
        ax.spy(sampled_adj, markersize=1, color='steelblue')
        ax.set_title(f'{title}\n')
    else:
        ax.spy(adj_matrix, markersize=1, color='steelblue')
        # ax.set_title(title)

    ax.set_xlabel('Node Index')
    ax.set_ylabel('Node Index')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_degree_distribution(
    data: torch_geometric.data.Data,
    title: str = "Node Degree Distribution",
    figsize: tuple = (8, 6),
    save_path: str = None
) -> None:
    """
    Plot the degree distribution as a separate figure.

    Args:
        data: PyTorch Geometric Data object containing edge_index
        title: Title for the plot
        figsize: Figure size as (width, height) tuple
        save_path: Path to save the figure
    """
    import numpy as np

    edge_index = data.edge_index.cpu().numpy()

    if hasattr(data, 'num_nodes') and data.num_nodes is not None:
        num_nodes = data.num_nodes
    else:
        num_nodes = int(edge_index.max()) + 1

    degrees = np.bincount(edge_index[0], minlength=num_nodes)

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(degrees, bins=min(50, num_nodes),
            color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.2)

    ax.set_xlabel('Node Degree')
    ax.set_ylabel('Frequency')
    # ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)

    stats_text = f'Mean: {avg_degree:.2f}\nMax: {max_degree}\nMin: {min_degree}'
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_graph_statistics(
    data: torch_geometric.data.Data,
    title: str = "Graph Statistics",
    figsize: tuple = (8, 6),
    save_path: str = None
) -> None:
    """
    Create a table visualization of graph statistics as a separate figure.

    Args:
        data: PyTorch Geometric Data object containing edge_index
        title: Title for the plot
        figsize: Figure size as (width, height) tuple
        save_path: Path to save the figure
    """
    import numpy as np

    edge_index = data.edge_index.cpu().numpy()

    if hasattr(data, 'num_nodes') and data.num_nodes is not None:
        num_nodes = data.num_nodes
    else:
        num_nodes = int(edge_index.max()) + 1

    num_edges = edge_index.shape[1]
    total_possible_edges = num_nodes * num_nodes
    sparsity = 1 - (num_edges / total_possible_edges)
    density = num_edges / total_possible_edges
    is_graph_undirected = is_undirected(data.edge_index)

    degrees = np.bincount(edge_index[0], minlength=num_nodes)
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    table_data = [
        ['Property', 'Value'],
        ['Nodes', f'{num_nodes:,}'],
        ['Edges', f'{num_edges:,}'],
        ['Undirected', str(is_graph_undirected)],
        ['', ''],
        ['Sparsity', f'{sparsity:.4%}'],
        ['Density', f'{density:.4%}'],
        ['Total Possible Edges', f'{total_possible_edges:,}'],
        ['', ''],
        ['Average Degree', f'{avg_degree:.2f}'],
        ['Max Degree', f'{max_degree}'],
        ['Min Degree', f'{min_degree}'],
    ]

    if hasattr(data, 'x') and data.x is not None:
        table_data.append(['Node Features', f'{data.x.shape[1]}'])
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        table_data.append(['Edge Features', f'{data.edge_attr.shape[1]}'])
    if hasattr(data, 'y') and data.y is not None:
        if data.y.ndim > 1:
            table_data.append(['Target Dimension', f'{data.y.shape[1]}'])
        else:
            table_data.append(['Target Dimension', '1'])

    table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.5, 0.5])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style alternating rows
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#E7E6E6')

    # ax.set_title(title, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_graph_sparsity(
    data: torch_geometric.data.Data,
    title: str = "Graph",
    save_path: str = "graph"
) -> None:
    """
    Generate three separate publication-quality plots for graph visualization:
    1. Adjacency matrix structure
    2. Node degree distribution
    3. Graph statistics table

    Args:
        data: PyTorch Geometric Data object containing edge_index
        title: Base title for the plots
        save_path: Base path for saving figures (without extension)

    Example:
        >>> from torch_geometric.data import Data
        >>> data = Data(edge_index=edge_index, num_nodes=100)
        >>> plot_graph_sparsity(data, title="My Graph", save_path="graph")
        >>> # Creates: graph_adjacency.png, graph_degree_dist.png, graph_statistics.png
    """
    import os

    base_name = os.path.splitext(save_path)[0]
    
    plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
    })

    # Plot 1: Adjacency matrix
    adj_path = f"{base_name}_adjacency.png"
    plot_adjacency_matrix(data, title=f"{title} - Adjacency Matrix", save_path=adj_path)

    # Plot 2: Degree distribution
    deg_path = f"{base_name}_degree_dist.png"
    plot_degree_distribution(data, title=f"{title} - Degree Distribution", save_path=deg_path)

    # Plot 3: Statistics
    stats_path = f"{base_name}_statistics.png"
    plot_graph_statistics(data, title=f"{title} - Statistics", save_path=stats_path)

    print(f"\nAll figures saved with prefix: {base_name}")
