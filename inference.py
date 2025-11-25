import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from torch_geometric.loader import DataLoader
from utils import calculate_aero_coefficients_2d, calculate_aero_coefficients_3d, find_latest_training_run
from train_utils import create_model
import argparse
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import datetime
from sklearn.metrics import r2_score

# Configure matplotlib for better plots
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

    # Check if BSMS model and wrap dataset
    model_name = params['model']['name']
    if model_name == 'bsms_mgn':
        from models.bsms_dataset_wrapper import prepare_bsms_data, BSMSDataLoader
        num_levels = params['model'].get('num_levels', 3)
        test_set = prepare_bsms_data(test_set, num_levels=num_levels)
    elif model_name == 'graphspectral_transolver':
        from models.graphSpectralTransolver import add_spectral_features_to_dataset
        print("\n=== Precomputing Spectral Features for Inference ===")
        spectral_dim = params['model'].get('spectral_dim', 8)
        laplacian_norm = params['model'].get('laplacian_norm', 'sym')
        test_set = add_spectral_features_to_dataset(test_set, spectral_dim, laplacian_norm, verbose=True)
    elif model_name == 'graphdistance_transolver':
        from models.graphDistanceTransolver import add_graph_distances_to_dataset
        print("\n=== Precomputing Graph Distances for Inference ===")
        max_hops = params['model'].get('max_hops', 5)
        test_set = add_graph_distances_to_dataset(test_set, max_hops, verbose=True)

    # Load model architecture and weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get dimensions from test data
    if model_name == 'bsms_mgn':
        sample_loader = BSMSDataLoader(test_set, batch_size=1, shuffle=False)
    else:
        sample_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    sample_batch = next(iter(sample_loader))
    input_node_dim = sample_batch.x.shape[1]
    input_edge_dim = sample_batch.edge_attr.shape[1]
    output_node_dim = sample_batch.y.shape[1]
    pos_dim = sample_batch.pos.shape[1] if hasattr(sample_batch, 'pos') else 2
    
    # Recreate model using factory function
    model_config = params['model']
    model = create_model(
        model_config=model_config,
        input_node_dim=input_node_dim,
        input_edge_dim=input_edge_dim,
        output_node_dim=output_node_dim,
        pos_dim=pos_dim
    )
    
    # Load weights
    weights_path = os.path.join(training_output_dir, "model_weights.pt")
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

    # Determine AMP settings based on training precision
    precision = params.get('training', {}).get('precision', 'float32').lower()
    use_amp = False
    amp_dtype = None
    if precision in ['bf16', 'bfloat16']:
        use_amp = True
        amp_dtype = torch.bfloat16

    return model, norm_stats, test_set, params, device, use_amp, amp_dtype

class AeroInference:
    """Comprehensive inference class for aerodynamic predictions."""

    def __init__(self, model, norm_stats: Dict, device: torch.device, params: Dict, use_amp: bool = False, amp_dtype: Optional[torch.dtype] = None):
        self.model = model.to(device)
        self.norm_stats = norm_stats
        self.device = device
        self.params = params
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.model.eval()

        # Move normalization stats to device
        self.device_norm_stats = {}
        for key, value in norm_stats.items():
            if isinstance(value, torch.Tensor):
                self.device_norm_stats[key] = value.to(device)
            else:
                self.device_norm_stats[key] = value
    
    @torch.no_grad()
    def predict_single(self, data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict for a single data sample."""
        data = data.to(self.device)

        # Check if model needs batch parameter (for poolMGN, MeshGraphNet_v2)
        model_class = self.model.__class__.__name__

        # Determine device type for autocast
        device_type = 'cuda' if self.device.type == 'cuda' else 'cpu'

        # Use autocast if AMP is enabled
        if self.use_amp and self.amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=self.amp_dtype):
                if model_class == 'BSMS_MeshGraphNet':
                    # BSMS model needs multi_data dict
                    multi_data = {}
                    for key, value in data.multi_data.items():
                        if isinstance(value, list):
                            multi_data[key] = [v.to(self.device) if torch.is_tensor(v) else v for v in value]
                        else:
                            multi_data[key] = value.to(self.device) if torch.is_tensor(value) else value
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, multi_data)

                elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                    # For single graph inference, create a batch tensor of zeros
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

                elif model_class in ['MLPNet']:
                    pred_scaled = self.model(data.x)

                elif model_class == 'GraphSpectralTransolver':
                    # GraphSpectralTransolver uses batch tensor and requires spectral_features
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    spectral_features = data.spectral_features if hasattr(data, 'spectral_features') else None
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch, spectral_features)

                elif model_class == 'GraphDistanceTransolver':
                    # GraphDistanceTransolver uses batch tensor and requires graph_distances
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    graph_distances = data.graph_distances if hasattr(data, 'graph_distances') else None
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch, graph_distances)

                elif model_class == 'TransolverAero':
                    # Transolver uses batch tensor for PyG batching
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)
                    
                elif model_class == 'MGNTransolver':
                    # Transolver uses batch tensor for PyG batching
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

                elif model_class == 'poolMGNTransolver':
                    # poolMGNTransolver uses batch tensor for PyG batching
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

                elif model_class == 'FourierMGNTransolver':
                    # FourierMGNTransolver uses batch tensor for PyG batching
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

                elif model_class == 'GCNTransolver':
                    # GCNTransolver uses batch tensor for PyG batching
                    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                    pred_scaled = self.model(data.x, data.edge_index, batch)

                elif model_class == 'GCN':
                    # GCN only needs node features and edge_index
                    pred_scaled = self.model(data.x, data.edge_index)

                else:
                    pred_scaled = self.model(data.x, data.edge_attr, data.edge_index)
        else:
            # No AMP, regular forward pass
            if model_class == 'BSMS_MeshGraphNet':
                # BSMS model needs multi_data dict
                multi_data = {}
                for key, value in data.multi_data.items():
                    if isinstance(value, list):
                        multi_data[key] = [v.to(self.device) if torch.is_tensor(v) else v for v in value]
                    else:
                        multi_data[key] = value.to(self.device) if torch.is_tensor(value) else value
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, multi_data)

            elif model_class in ['poolMGN', 'MeshGraphNet_v2']:
                # For single graph inference, create a batch tensor of zeros
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

            elif model_class in ['MLPNet']:
                pred_scaled = self.model(data.x)

            elif model_class == 'GraphSpectralTransolver':
                # GraphSpectralTransolver uses batch tensor and requires spectral_features
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                spectral_features = data.spectral_features if hasattr(data, 'spectral_features') else None
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch, spectral_features)

            elif model_class == 'GraphDistanceTransolver':
                # GraphDistanceTransolver uses batch tensor and requires graph_distances
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                graph_distances = data.graph_distances if hasattr(data, 'graph_distances') else None
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch, graph_distances)

            elif model_class == 'TransolverAero':
                # Transolver uses batch tensor for PyG batching
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

            elif model_class == 'MGNTransolver':
                # MGNTransolver uses batch tensor for PyG batching
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

            elif model_class == 'poolMGNTransolver':
                # poolMGNTransolver uses batch tensor for PyG batching
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

            elif model_class == 'FourierMGNTransolver':
                # FourierMGNTransolver uses batch tensor for PyG batching
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)

            elif model_class == 'GCNTransolver':
                # GCNTransolver uses batch tensor for PyG batching
                batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
                pred_scaled = self.model(data.x, data.edge_index, batch)

            elif model_class == 'GCN':
                # GCN only needs node features and edge_index
                pred_scaled = self.model(data.x, data.edge_index)

            else:
                pred_scaled = self.model(data.x, data.edge_attr, data.edge_index)

        # Denormalize predictions
        pred_unscaled = (pred_scaled * self.device_norm_stats['target_std'] +
                        self.device_norm_stats['target_mean']).cpu()

        # Denormalize ground truth
        target_unscaled = (data.y * self.device_norm_stats['target_std'] +
                          self.device_norm_stats['target_mean']).cpu()

        # Keep scaled versions for training-comparable errors
        pred_scaled = pred_scaled.cpu()
        target_scaled = data.y.cpu()

        return pred_unscaled, target_unscaled, pred_scaled, target_scaled
    
    def compute_errors(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """Compute various error metrics."""
        mae = torch.mean(torch.abs(pred - target)).item()
        mse = torch.mean((pred - target) ** 2).item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # Relative errors (avoid division by zero)
        target_nonzero = target[torch.abs(target) > 1e-8]
        pred_nonzero = pred[torch.abs(target) > 1e-8]
        if len(target_nonzero) > 0:
            relative_mae = torch.mean(torch.abs((pred_nonzero - target_nonzero) / target_nonzero)).item()
            relative_rmse = torch.sqrt(torch.mean(((pred_nonzero - target_nonzero) / target_nonzero) ** 2)).item()
        else:
            relative_mae = relative_rmse = float('nan')
        
        return {
            'mae': mae,
            'mse': mse, 
            'rmse': rmse,
            'relative_mae': relative_mae,
            'relative_rmse': relative_rmse
        }
    
    def compute_rrmse_percent(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute relative RMSE as percentage (mean across all features)."""
        # Compute RMSE for each feature
        # feature_rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
        # feature_mean_abs = torch.mean(torch.abs(target), dim=0)
        
        # # Relative RMSE per feature (avoid division by zero)
        # feature_rrmse = torch.where(feature_mean_abs > 1e-8, 
        #                            feature_rmse / feature_mean_abs, 
        #                            torch.zeros_like(feature_rmse))
        
        # # Mean relative RMSE across all features as percentage
        # mean_rrmse_percent = torch.mean(feature_rrmse).item() * 100
        # return mean_rrmse_percent
        return (
            torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)
        ).mean().item() * 100
    
    def plot_2d_airfoil_predictions(self, data, pred: torch.Tensor, target: torch.Tensor,
                                save_path: str, case_name: str = ""):
        """Create separate plots for predictions."""
        # With AMP, data stays in float32/float64, so no need for dtype checks
        pos = data.pos.cpu().numpy()
        x_coords = pos[:, 0]
        y_coords = pos[:, 1]
        
        # Get target features from dataset config or use fallback
        target_features = self.params.get('dataset', {}).get('output_features', [f'feature_{i}' for i in range(target.shape[1])])
        n_features = len(target_features)
        
        # Create base path without extension for multiple files
        base_path = save_path.rsplit('.', 1)[0]
        # For prefix naming, extract directory and filename separately
        dir_path = os.path.dirname(save_path)
        filename_base = os.path.basename(save_path).rsplit('.', 1)[0]
        
        # 1. PREDICTIONS PLOT
        fig_pred = plt.figure(figsize=(12, 4 * n_features))
        for i, feature_name in enumerate(target_features):
            # With AMP, data stays in float32/float64, so no need for dtype checks
            pred_feature = pred[:, i].numpy()
            target_feature = target[:, i].numpy()

            ax = plt.subplot(n_features, 1, i + 1)
            scatter1 = plt.scatter(x_coords, target_feature, c='b', 
                                alpha=0.7, s=20, label='Ground Truth', marker='o')
            scatter2 = plt.scatter(x_coords, pred_feature, c='g', 
                                alpha=0.7, s=20, marker='^', label='Prediction')
            plt.xlabel('X Coordinate')
            plt.ylabel(f'{feature_name}')
            plt.title(f'{feature_name} vs X-coordinate')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
        
        plt.suptitle(f'Predictions Comparison - {case_name}', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{base_path}_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_ahmedBody_vtu(self, data, pred: torch.Tensor, 
                                     original_file_path: str, output_path: str):
        """Export 3D VTU/VTP file with predictions included."""
        try:
            # Load original mesh
            mesh = pv.read(original_file_path)
            
            if mesh is None:
                raise ValueError(f"Could not load mesh from {original_file_path}")
            
            # If it's a volume mesh, extract surface
            if hasattr(mesh, 'n_cells') and mesh.n_cells > 0:
                try:
                    cell = mesh.get_cell(0)
                    if hasattr(cell, 'type') and cell.type != pv.CellType.TRIANGLE:
                        mesh = mesh.extract_surface()
                except:
                    # If we can't determine cell type, just continue with original mesh
                    pass
            
            # Verify mesh has point_data attribute
            if not hasattr(mesh, 'point_data'):
                raise ValueError("Mesh does not have point_data attribute")
            
            
            # Add predictions to mesh
            mesh.point_data['p_pred'] = pred[:, 0].numpy()
            p_error = mesh.point_data['p'] - pred[:, 0].numpy()
            mesh.point_data['p_error'] = p_error
            mesh.point_data['wallShearStress_pred'] = pred[:, 1:4].numpy()
            shear_error = mesh.point_data['wallShearStress'] - pred[:, 1:4].numpy()
            mesh.point_data['wallShearStress_error'] = shear_error
            
            
            # Save mesh
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if hasattr(mesh, 'save'):
                mesh.save(output_path)
                return True
            else:
                raise ValueError("Mesh does not have save method")
            
        except Exception as e:
            print(f"Warning: Could not export VTU for {original_file_path}: {e}")
            return False

    def pv_plot_ahmedBody(self, original_file_path: str, output_path: str):
        """Create PyVista plot of Ahmed body with predictions."""
        try:
            # Load original mesh
            mesh = pv.read(original_file_path)
            camera_position = [(-2.50, 0.9, 0.65), (-0.6, -0.02, 0.1215), (0.2, -0.12, 0.9685)]
            
            
            if mesh is None:
                raise ValueError(f"Could not load mesh from {original_file_path}")
            
            # Verify mesh has point_data attribute
            if not hasattr(mesh, 'point_data'):
                raise ValueError("Mesh does not have point_data attribute")
            
            # Get individual scalar ranges for each subplot independently
            # This ensures subplots don't affect each other's coloring

            # Pressure: use common range for pred and true (for comparison)
            p_min = min(mesh["p_pred"].min(), mesh["p"].min())
            p_max = max(mesh["p_pred"].max(), mesh["p"].max())

            # Pressure error: independent range, symmetric around zero
            p_err_abs_max = max(abs(mesh["p_error"].min()), abs(mesh["p_error"].max()))

            # Shear stress magnitude: use common range for pred and true
            s_min = min(mesh["wallShearStress_pred"].min(), mesh["wallShearStress"].min())
            s_max = max(mesh["wallShearStress_pred"].max(), mesh["wallShearStress"].max())

            # Shear stress error magnitude: independent range, symmetric around zero
            s_err_abs_max = max(abs(mesh["wallShearStress_error"].min()), abs(mesh["wallShearStress_error"].max()))

            plotter = pv.Plotter(shape=(2, 3), window_size=(1800, 1200), off_screen=True)

            plotter.subplot(0, 0)
            plotter.add_mesh(mesh, scalars="p_pred", clim=[p_min, p_max],
                           cmap='viridis', show_scalar_bar=True,
                           scalar_bar_args={'title': 'p_pred'}, copy_mesh=True)
            plotter.add_text("Prediction: Pressure", position='upper_left', font_size=12)
            plotter.camera_position = camera_position

            plotter.subplot(0, 1)
            plotter.add_mesh(mesh, scalars="p", clim=[p_min, p_max],
                           cmap='viridis', show_scalar_bar=True,
                           scalar_bar_args={'title': 'p'}, copy_mesh=True)
            plotter.add_text("Ground Truth: Pressure", position='upper_left', font_size=12)
            plotter.camera_position = camera_position

            plotter.subplot(0, 2)
            plotter.add_mesh(mesh, scalars="p_error",
                           clim=[-p_err_abs_max, p_err_abs_max],
                           cmap='coolwarm', show_scalar_bar=True,
                           scalar_bar_args={'title': 'p_error'}, copy_mesh=True)
            plotter.add_text("Error: Pressure", position='upper_left', font_size=12)
            plotter.camera_position = camera_position

            plotter.subplot(1, 0)
            plotter.add_mesh(mesh, scalars="wallShearStress_pred",
                           clim=[s_min, s_max], cmap='viridis', show_scalar_bar=True,
                           scalar_bar_args={'title': 'tau_mag_pred'}, copy_mesh=True)
            plotter.add_text("Prediction: Wall Shear Stress Mag", position='upper_left', font_size=12)
            plotter.camera_position = camera_position

            plotter.subplot(1, 1)
            plotter.add_mesh(mesh, scalars="wallShearStress",
                           clim=[s_min, s_max], cmap='viridis', show_scalar_bar=True,
                           scalar_bar_args={'title': 'tau_mag'}, copy_mesh=True)
            plotter.add_text("Ground Truth: Wall Shear Stress Mag", position='upper_left', font_size=12)
            plotter.camera_position = camera_position

            plotter.subplot(1, 2)
            plotter.add_mesh(mesh, scalars="wallShearStress_error",
                           clim=[-s_err_abs_max, s_err_abs_max],
                           cmap='coolwarm', show_scalar_bar=True,
                           scalar_bar_args={'title': 'tau_mag_error'}, copy_mesh=True)
            plotter.add_text("Error: Wall Shear Stress Mag", position='upper_left', font_size=12)
            plotter.camera_position = camera_position
            
            plotter.link_views()
            plotter.screenshot(output_path)
            return True
            
        except Exception as e:
            print(f"Warning: Could not create PyVista plot for {original_file_path}: {e}")
            return False
            

    def plot_aero_coefficients_r2(self, aero_coeffs: Dict, output_dir: str, dataset_name: str):
        """Create R² scatter plots with fit lines for aerodynamic coefficients."""

        # Determine which coefficients to plot based on dataset
        if dataset_name == "airfoil_2d":
            coeff_names = ['CA', 'CN', 'Cm']
        elif dataset_name == "ahmed_body":
            coeff_names = ['CA']
        else:
            print("No aerodynamic coefficients to plot for this dataset type.")
            return

        # Filter out empty coefficient lists
        coeffs_to_plot = []
        for coeff in coeff_names:
            if len(aero_coeffs[f'{coeff}_pred']) > 0:
                coeffs_to_plot.append(coeff)

        if not coeffs_to_plot:
            print("No aerodynamic coefficients available to plot.")
            return

        n_coeffs = len(coeffs_to_plot)
        fig, axes = plt.subplots(1, n_coeffs, figsize=(6 * n_coeffs, 5))

        # Handle single coefficient case (axes won't be an array)
        if n_coeffs == 1:
            axes = [axes]

        for idx, coeff_name in enumerate(coeffs_to_plot):
            pred_values = np.array(aero_coeffs[f'{coeff_name}_pred'])
            true_values = np.array(aero_coeffs[f'{coeff_name}_true'])

            # Compute R² score
            r2 = r2_score(true_values, pred_values)

            # Compute linear fit
            # coeffs = np.polyfit(true_values, pred_values, 1)
            # poly = np.poly1d(coeffs)

            # Create fit line points
            min_val = min(true_values.min(), pred_values.min())
            max_val = max(true_values.max(), pred_values.max())
            # fit_line = np.linspace(min_val, max_val, 100)

            # Plot
            ax = axes[idx]
            ax.scatter(true_values, pred_values, alpha=0.6, s=50, edgecolors='k', linewidths=0.5)
            # ax.plot(fit_line, poly(fit_line), 'r--', linewidth=2, label=f'Fit: y={coeffs[0]:.3f}x+{coeffs[1]:.3f}')
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', linewidth=1, alpha=0.5)

            ax.set_xlabel(f'True {coeff_name}', fontsize=14)
            ax.set_ylabel(f'Predicted {coeff_name}', fontsize=14)
            ax.set_title(f'{coeff_name}: R² = {r2:.4f}', fontsize=16, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()

        # Save plot
        plot_path = os.path.join(output_dir, 'aerodynamic_coefficients_r2.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nAerodynamic Coefficient R² Analysis:")
        for coeff_name in coeffs_to_plot:
            pred_values = np.array(aero_coeffs[f'{coeff_name}_pred'])
            true_values = np.array(aero_coeffs[f'{coeff_name}_true'])
            r2 = r2_score(true_values, pred_values)
            print(f"  {coeff_name}: R² = {r2:.4f}")
        print(f"  Plot saved to: {plot_path}")

    def run_inference(self, test_dataset, output_dir: str, original_data_dir: Optional[str] = None):
        """Run comprehensive inference on test dataset."""
        print(f"Running inference on {len(test_dataset)} test cases...")
        
        # Create output directories
        date_time = datetime.datetime.now().strftime("%d-%m_%H-%M") 
        inference_dir = os.path.join(output_dir, f"inference_results_{date_time}")
        plots_dir = os.path.join(inference_dir, "plots")
        vtu_dir = os.path.join(inference_dir, "vtu_exports") 
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(vtu_dir, exist_ok=True)
        
        # Results storage
        all_case_errors = []
        # Get target features from dataset config or use fallback
        target_features = self.params.get('dataset', {}).get('output_features', [f'feature_{i}' for i in range(test_dataset[0].y.shape[1])])
        
        dataset_name = self.params.get('dataset', {}).get('name', 'dataset')
        data_dir = self.params.get('dataset', {}).get('data_dir', '')
                
        # Collect all predictions for comprehensive analysis
        all_pred_phys = []
        all_target_phys = []
        all_pred_norm = []
        all_target_norm = []

        # Collect aerodynamic coefficients for R² analysis
        aero_coeffs = {
            'CA_pred': [], 'CA_true': [],
            'CN_pred': [], 'CN_true': [],
            'Cm_pred': [], 'Cm_true': []
        }

        for i, data in enumerate(test_dataset):
            pred_phys, target_phys, pred_norm, target_norm = self.predict_single(data)
            
            # Store for comprehensive analysis
            all_pred_phys.append(pred_phys)
            all_target_phys.append(target_phys)
            all_pred_norm.append(pred_norm)
            all_target_norm.append(target_norm)
            
            # Compute RRMSE for this case
            case_rrmse = self.compute_rrmse_percent(pred_phys, target_phys)
            
            
            coeff_str = ""
            if dataset_name == "airfoil_2d":
                mach = data.mach
                alpha = data.alpha
                
                pred_pressure = pred_phys[:, 0:1]
                pred_shear = pred_phys[:, 1:3]
                true_pressure = target_phys[:, 0:1]
                true_shear = target_phys[:, 1:3]
                
                
                true_coeffs = calculate_aero_coefficients_2d(
                                data,
                                pressure=true_pressure,
                                shear_stress=true_shear,
                                reference_area=1e-2,
                                reference_length=1.0,
                                dynamic_pressure=0.5 * 1.4 * 101325 * mach * mach
                               )

                    # Calculate coefficients for predictions
                pred_coeffs = calculate_aero_coefficients_2d(
                            data,
                            pressure=pred_pressure,
                            shear_stress=pred_shear,
                            reference_area=1e-2,
                            reference_length=1.0,
                            dynamic_pressure=0.5 * 1.4 * 101325 * mach * mach
                            )

                # Store coefficients for R² analysis
                aero_coeffs['CA_pred'].append(pred_coeffs['CA'])
                aero_coeffs['CA_true'].append(true_coeffs['CA'])
                aero_coeffs['CN_pred'].append(pred_coeffs['CN'])
                aero_coeffs['CN_true'].append(true_coeffs['CN'])
                aero_coeffs['Cm_pred'].append(pred_coeffs['Cm'])
                aero_coeffs['Cm_true'].append(true_coeffs['Cm'])

                coeff_str = (f" | CA:{pred_coeffs['CA']:7.4f} ({true_coeffs['CA']:7.4f}) "
                            f"| CN:{pred_coeffs['CN']:7.4f} ({true_coeffs['CN']:7.4f}) "
                            f"| Cm:{pred_coeffs['Cm']:7.4f} ({true_coeffs['Cm']:7.4f})")

                print(f"Error in case{i:03d}: {case_rrmse:7.4f}%{coeff_str}")


            elif dataset_name == "ahmed_body":

                velocity = data.Velocity
                height = data.Height
                width = data.Width

                pred_pressure = pred_phys[:, 0:1]
                pred_shear = pred_phys[:, 1:4]
                true_pressure = target_phys[:, 0:1]
                true_shear = target_phys[:, 1:4]

                mesh = pv.read(os.path.join(data_dir, data.split, data.case_no+'.vtp'))
                surface = mesh.extract_surface()
                surface = surface.compute_normals(cell_normals=True, point_normals=False, consistent_normals=True, inplace=False)
                surface = surface.compute_cell_sizes(length=False, area=True, volume=False)
                
                node_areas = surface.cell_data['Area']
                normals = surface.cell_data["Normals"]

                surface.point_data["p_pred"] = pred_pressure
                surface.point_data["wallShearStress_pred"] = pred_shear

                surface = surface.point_data_to_cell_data(pass_point_data=False)

                coeffs = calculate_aero_coefficients_3d(
                    surface,
                    reference_area=height * width * 1e-6 / 2,
                    reference_length= 1.0,
                    dynamic_pressure= 0.5 * 1.225 * velocity * velocity
                )

                # Store coefficients for R² analysis
                aero_coeffs['CA_pred'].append(coeffs['CA_pred'])
                aero_coeffs['CA_true'].append(coeffs['CA_true'])

                coeff_str = (f" | CA:{coeffs['CA_pred']:7.4f} ({coeffs['CA_true']:7.4f})")

                print(f"Error in case{i:03d}: {case_rrmse:7.4f}%{coeff_str}")
                    
            
            
            # Compute feature-wise errors for this case (both scales)
            case_errors_phys = {}
            case_errors_norm = {}
            
            for j, feature_name in enumerate(target_features):
                # Physical scale errors
                mae_phys = torch.mean(torch.abs(pred_phys[:, j] - target_phys[:, j])).item()
                mse_phys = torch.mean((pred_phys[:, j] - target_phys[:, j])**2).item()
                case_errors_phys[feature_name] = {'mae': mae_phys, 'mse': mse_phys}
                
                # Normalized scale errors
                mae_norm = torch.mean(torch.abs(pred_norm[:, j] - target_norm[:, j])).item()
                mse_norm = torch.mean((pred_norm[:, j] - target_norm[:, j])**2).item()
                case_errors_norm[feature_name] = {'mae': mae_norm, 'mse': mse_norm}
            
            # Store case error information
            case_error_info = {
                'case_id': i,
                'rrmse_percent': case_rrmse,
                'errors_physical': case_errors_phys,
                'errors_normalized': case_errors_norm,
                'coeff_str': coeff_str  # Store coefficient string for this case
            }
            
            
            # Add case-specific information if available
            if hasattr(data, 'airfoil'):
                case_error_info['airfoil'] = data.airfoil
            if hasattr(data, 'mach'):
                case_error_info['mach'] = data.mach.item() if torch.is_tensor(data.mach) else data.mach
            if hasattr(data, 'alpha'):
                case_error_info['alpha'] = data.alpha.item() if torch.is_tensor(data.alpha) else data.alpha
            if hasattr(data, 'case_no'):
                case_error_info['case_no'] = data.case_no.item() if torch.is_tensor(data.case_no) else data.case_no

            all_case_errors.append(case_error_info)
            
            # Generate visualizations based on dimension
            if dataset_name == "airfoil_2d":
                # 2D airfoil plotting
                case_name = f"Case {i:03d}"
                if hasattr(data, 'airfoil'):
                    case_name += f" - {data.airfoil}"
                if hasattr(data, 'mach') and hasattr(data, 'alpha'):
                    mach_val = data.mach.item() if torch.is_tensor(data.mach) else data.mach
                    alpha_val = data.alpha.item() if torch.is_tensor(data.alpha) else data.alpha
                    case_name += f" (M={mach_val:.2f}, α={alpha_val:.1f}°)"
                
                plot_path = os.path.join(plots_dir, f"prediction_case_{i:03d}.png")
                self.plot_2d_airfoil_predictions(data, pred_phys, target_phys, plot_path, case_name)
            
            elif dataset_name == "ahmed_body":
                # 3D VTU export

                original_file = os.path.join(data_dir, data.split, data.case_no+'.vtp')
                output_file = os.path.join(vtu_dir, f"{data.case_no}_predictions.vtp")
                self.export_ahmedBody_vtu(data, pred_phys, original_file, output_file)
                png_file = os.path.join(vtu_dir, f"{data.case_no}_predictions.png")
                self.pv_plot_ahmedBody(output_file, png_file)

        # Plot R² scatter plots for aerodynamic coefficients
        self.plot_aero_coefficients_r2(aero_coeffs, inference_dir, dataset_name)

        # Compute test-set mean feature-wise errors
        pred_phys_all = torch.cat(all_pred_phys, dim=0)
        target_phys_all = torch.cat(all_target_phys, dim=0)
        pred_norm_all = torch.cat(all_pred_norm, dim=0)
        target_norm_all = torch.cat(all_target_norm, dim=0)
        
        test_mean_errors_phys = {}
        test_mean_errors_norm = {}
        
        for j, feature_name in enumerate(target_features):
            # Physical scale test-set mean
            mae_phys = torch.mean(torch.abs(pred_phys_all[:, j] - target_phys_all[:, j])).item()
            mse_phys = torch.mean((pred_phys_all[:, j] - target_phys_all[:, j])**2).item()
            test_mean_errors_phys[feature_name] = {'mae': mae_phys, 'mse': mse_phys}
            
            # Normalized scale test-set mean
            mae_norm = torch.mean(torch.abs(pred_norm_all[:, j] - target_norm_all[:, j])).item()
            mse_norm = torch.mean((pred_norm_all[:, j] - target_norm_all[:, j])**2).item()
            test_mean_errors_norm[feature_name] = {'mae': mae_norm, 'mse': mse_norm}
        
        # Create final errors structure
        final_errors = {
            'per_case_errors': all_case_errors,
            'test_set_mean': {
                'errors_physical': test_mean_errors_phys,
                'errors_normalized': test_mean_errors_norm
            }
        }
        
        # Save errors.txt
        errors_txt_path = os.path.join(inference_dir, "errors.txt")
        with open(errors_txt_path, 'w') as f:
            # Compute test set mean values across all targets
            test_mean_nmae = np.mean([test_mean_errors_norm[f]['mae'] for f in target_features])
            test_mean_nmse = np.mean([test_mean_errors_norm[f]['mse'] for f in target_features])
            test_mean_mae = np.mean([test_mean_errors_phys[f]['mae'] for f in target_features])
            test_mean_mse = np.mean([test_mean_errors_phys[f]['mse'] for f in target_features])
            
            # Compute mean RRMSE across all cases
            test_mean_rrmse = np.mean([case['rrmse_percent'] for case in all_case_errors])
            
            # Write test set mean at the top
            f.write(f"TEST_MEAN | rrmse:{test_mean_rrmse:6.2f} | nmae:{test_mean_nmae:8.6f} | nmse:{test_mean_nmse:8.6f} | mae:{test_mean_mae:7.2f} | mse:{test_mean_mse:12.2f}\n")
            f.write("\n")
            
            # Write each case
            for case_info in all_case_errors:
                # Compute mean across all targets for this case
                case_nmae = np.mean([case_info['errors_normalized'][f]['mae'] for f in target_features])
                case_nmse = np.mean([case_info['errors_normalized'][f]['mse'] for f in target_features])
                case_mae = np.mean([case_info['errors_physical'][f]['mae'] for f in target_features])
                case_mse = np.mean([case_info['errors_physical'][f]['mse'] for f in target_features])
                
                if dataset_name == "airfoil_2d":
                    airfoil = case_info.get('airfoil', 'N/A')
                    mach = case_info.get('mach', 'N/A')
                    alpha = case_info.get('alpha', 'N/A')
                    case_coeff_str = case_info.get('coeff_str', '')  # Get coefficient string for this case
                    
                    # Format numbers
                    if isinstance(mach, (int, float)):
                        mach = f"{mach:.2f}"
                    if isinstance(alpha, (int, float)):
                        alpha = f"{alpha:.2f}"
                    
                    # Write line with fixed-width formatting
                    line = f"case_{case_info['case_id']:03d} | rrmse:{case_info['rrmse_percent']:6.2f} | nmae:{case_nmae:8.6f} | nmse:{case_nmse:8.6f} | mae:{case_mae:7.2f} | mse:{case_mse:12.2f}{case_coeff_str} | {airfoil:8s} | {str(mach):4s} | {str(alpha):5s}"
                    f.write(line + "\n")
                    
                elif dataset_name == "ahmed_body":
                    case_no = case_info.get('case_no', 'N/A')
                    case_coeff_str = case_info.get('coeff_str', '')  # Get coefficient string for this case
                    
                    # Write line with fixed-width formatting
                    line = f"case_{case_info['case_id']:03d} | rrmse:{case_info['rrmse_percent']:6.2f} | nmae:{case_nmae:8.6f} | nmse:{case_nmse:8.6f} | mae:{case_mae:7.2f} | mse:{case_mse:12.2f}{case_coeff_str} | {str(case_no):5s}"
                    f.write(line + "\n")
                    
        #save target specific errors as json
        errors_json_path = os.path.join(inference_dir, "errors_target.json")
        with open(errors_json_path, 'w') as f:
            json.dump(final_errors, f, indent=4)
        
        print(f"Inference complete! Results saved to: {inference_dir}")
        return inference_dir

def main():
    parser = argparse.ArgumentParser(description="Run inference on trained aerodynamic GNN model")
    parser.add_argument("--training_dir", type=str, default=None,
                       help="Path to training output directory. If not provided, uses latest run.")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Original data directory for VTU export (optional)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use: 'cuda', 'cpu', or 'auto'")
    
    args = parser.parse_args()
    
    # Find training directory if not provided
    if args.training_dir is None:
        print("No training directory specified, looking for latest run...")
        try:
            args.training_dir = find_latest_training_run()
            print(f"Found latest training run: {args.training_dir}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify a training directory with --training_dir")
            sys.exit(1)
    
    # Validate training directory
    if not os.path.exists(args.training_dir):
        raise FileNotFoundError(f"Training directory not found: {args.training_dir}")
    
    required_files = ["model_weights.pt", "normalization_stats.pt", "experiment_params.json"]
    for file in required_files:
        if not os.path.exists(os.path.join(args.training_dir, file)):
            raise FileNotFoundError(f"Required file not found: {os.path.join(args.training_dir, file)}")
    
    print(f"Loading model and data from: {args.training_dir}")

    # Load everything
    model, norm_stats, test_set, params, device, use_amp, amp_dtype = load_model_and_data(args.training_dir)

    if args.device != "auto":
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Test set contains {len(test_set)} samples")

    if use_amp:
        print(f"Using automatic mixed precision with {amp_dtype}")

    # Create inference engine with AMP settings
    inference_engine = AeroInference(model, norm_stats, device, params, use_amp=use_amp, amp_dtype=amp_dtype)

    # Run inference
    inference_engine.run_inference(test_set, args.training_dir, args.data_dir)


if __name__ == "__main__":
    main()
