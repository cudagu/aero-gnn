import torch
import matplotlib.pyplot as plt
import os
import numpy as np

# A global dictionary to store our SVD logs
# We'll store logs in memory and save at the end
SVD_LOGS = {}

# --- List of layers we want to analyze ---
# --- (Update this list based on Step 1) ---
TARGET_LAYER_NAMES = ['node_encoder.layers.1',
    'layers.0.edge_block.mlp.1',
    'layers.0.node_block.mlp.layers.0',
    'layers.7.edge_block.mlp.1',
    'layers.7.node_block.mlp.layers.0',          
    'decoder.layers.0'
]

@torch.no_grad() # Crucial: we don't want to track gradients for this
def log_singular_values(model: torch.nn.Module, epoch: int):
    """
    Performs SVD on target layers and logs their singular values.
    """
    global SVD_LOGS
    
    # Switch to eval mode temporarily if model is in training mode
    was_training = model.training
    model.eval()
    
    if epoch == 0:
        # Initialize the log dictionary on the first epoch
        for name in TARGET_LAYER_NAMES:
            SVD_LOGS[name] = []
            
    for layer_name, module in model.named_modules():
        if layer_name in TARGET_LAYER_NAMES and isinstance(module, torch.nn.Linear):
            
            # Get the weight matrix, move to CPU, and convert to float32
            # for stable SVD
            W = module.weight.detach().cpu().to(torch.float32)
            
            # Perform SVD - we only need the singular values 'S'
            # torch.linalg.svdvals is much faster than full torch.linalg.svd
            S = torch.linalg.svdvals(W)
            
            # Store the singular values (as a numpy array for less memory)
            SVD_LOGS[layer_name].append(S.numpy())

    # Restore model's training state
    if was_training:
        model.train()

def save_svd_plots(log_dir: str):
    """
    Plots the SVD spectrum and rank evolution after training is complete.
    """
    global SVD_LOGS
    if not SVD_LOGS:
        print("No SVD logs to plot.")
        return

    print("Saving SVD analysis plots...")
    os.makedirs(log_dir, exist_ok=True)
    
    num_epochs = len(SVD_LOGS[TARGET_LAYER_NAMES[0]])
    #epochs are in mod 5 increments
    epochs = [5 * i for i in range(num_epochs)]
    
    for layer_name, logs in SVD_LOGS.items():
        
        # --- Plot 1: Singular Value Spectrum (Final Epoch) ---
        plt.figure(figsize=(12, 6))
        final_S = logs[-1]
        
        # Sort values and plot on a log scale
        sorted_S = np.sort(final_S)[::-1]
        plt.semilogy(sorted_S, 'b-o', markersize=3)
        plt.title(f"Singular Value Spectrum (Final Epoch)\nLayer: {layer_name}")
        plt.xlabel("Singular Value Index (Sorted)")
        plt.ylabel("Singular Value (Log Scale)")
        plt.grid(True, which="both", ls="--")
        plt.savefig(os.path.join(log_dir, f"{layer_name}_spectrum.png"))
        plt.close()
        
        # --- Plot 2: "Effective Rank" Evolution ---
        # We define "effective rank" as the number of singular values
        # needed to capture 99% of the matrix energy (Frobenius norm squared)
        
        effective_ranks = []
        for S_epoch in logs:
            S_squared = S_epoch**2
            total_energy = np.sum(S_squared)
            
            # Find cumulative energy
            cumulative_energy = np.cumsum(S_squared)
            
            # Find the index where we reach 99% of total energy
            try:
                # np.argmax returns the *first* index where condition is True
                rank = np.argmax(cumulative_energy >= total_energy * 0.99) + 1
            except ValueError:
                rank = len(S_epoch) # Should not happen
            
            effective_ranks.append(rank)
            
        plt.figure(figsize=(12, 6))
        plt.plot(epochs, effective_ranks, 'r-x')
        plt.title(f"Effective Rank (99% Energy) Evolution\nLayer: {layer_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Effective Rank")
        plt.grid(True, ls="--")
        plt.savefig(os.path.join(log_dir, f"{layer_name}_rank_evolution.png"))
        plt.close()

    print(f"SVD analysis plots saved to {log_dir}")