import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
from torch_geometric.loader import DataLoader
from utils import calculate_aero_coefficients_2d, calculate_aero_coefficients_3d, load_model_and_data, find_latest_training_run
import argparse
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import datetime

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

class AeroInference:
    """Comprehensive inference class for aerodynamic predictions."""
    
    def __init__(self, model, norm_stats: Dict, device: torch.device, params: Dict):
        self.model = model.to(device)
        self.norm_stats = norm_stats
        self.device = device
        self.params = params
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
        if model_class in ['poolMGN', 'MeshGraphNet_v2']:
            # For single graph inference, create a batch tensor of zeros
            batch = torch.zeros(data.x.size(0), dtype=torch.long, device=self.device)
            pred_scaled = self.model(data.x, data.edge_attr, data.edge_index, batch)
            
        elif model_class in ['MLPNet']:
            pred_scaled = self.model(data.x)
    
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
        feature_rmse = torch.sqrt(torch.mean((pred - target) ** 2, dim=0))
        feature_mean_abs = torch.mean(torch.abs(target), dim=0)
        
        # Relative RMSE per feature (avoid division by zero)
        feature_rrmse = torch.where(feature_mean_abs > 1e-8, 
                                   feature_rmse / feature_mean_abs, 
                                   torch.zeros_like(feature_rmse))
        
        # Mean relative RMSE across all features as percentage
        mean_rrmse_percent = torch.mean(feature_rrmse).item() * 100
        return mean_rrmse_percent
    
    def plot_2d_airfoil_predictions(self, data, pred: torch.Tensor, target: torch.Tensor, 
                                save_path: str, case_name: str = ""):
        """Create separate plots for predictions."""
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
    
    def export_3d_vtu_with_predictions(self, data, pred: torch.Tensor, 
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
            
            # Get target features from dataset config or use fallback
            target_features = self.params.get('dataset', {}).get('output_features', [f'feature_{i}' for i in range(pred.shape[1])])
            
            # Add predictions to mesh
            for i, feature_name in enumerate(target_features):
                if hasattr(mesh, 'point_data'):
                    mesh.point_data[f'predicted_{feature_name}'] = pred[:, i].numpy()
                    
                    # Add ground truth if available
                    if hasattr(data, 'y') and data.y is not None:
                        target = (data.y * self.device_norm_stats['target_std'] + 
                                 self.device_norm_stats['target_mean']).cpu()
                        mesh.point_data[f'true_{feature_name}'] = target[:, i].numpy()
                        
                        # Add error
                        error = pred[:, i].numpy() - target[:, i].numpy()
                        mesh.point_data[f'error_{feature_name}'] = error
            
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

                coeff_str = (f" | CA:{coeffs['CA_pred']:7.4f} ({coeffs['CA_true']:7.4f}) "
                            f"| CN:{coeffs['CN_pred']:7.4f} ({coeffs['CN_true']:7.4f}) "
                            f"| CY:{coeffs['CY_pred']:7.4f} ({coeffs['CY_true']:7.4f})")
                
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
                self.export_3d_vtu_with_predictions(data, pred_phys, original_file, output_file)
        
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
    model, norm_stats, test_set, params, device = load_model_and_data(args.training_dir)
    
    if args.device != "auto":
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Test set contains {len(test_set)} samples")
    
    # Create inference engine
    inference_engine = AeroInference(model, norm_stats, device, params)
    
    # Run inference
    inference_engine.run_inference(test_set, args.training_dir, args.data_dir)


if __name__ == "__main__":
    main()
