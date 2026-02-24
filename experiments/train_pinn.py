import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.PINN import PINN
from src.physics.PDEs import Poisson1D, Burgers1D
from src.utils.plotting import plot_1d_pinn

# =========================================================================
# This script provides a runnable entry point to test the standard PINN baseline.
# It uses the identical data and physics formulation as `train_binn.py` but optimizes
# the network conventionally via Adam, verifying that the `compute_loss` method works.
# =========================================================================

def run_pinn():
    # 1. Setup Data and Physics
    # True function: u(x) = sin(pi * x)
    # PDE: u_xx = -pi^2 * sin(pi * x)
    
    # solution and boundary data (x_u, y_u)
    x_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y_u = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    
    # Collocation points (x_f), forcing term measurements (y_f)
    x_f = torch.linspace(-1, 1, 20).view(-1, 1).requires_grad_(True)
    y_f = - (torch.pi ** 2) * torch.sin(torch.pi * x_f).detach()

    # Add some noise to the data
    y_u = y_u + torch.randn_like(y_u) * 0.1
    y_f = y_f + torch.randn_like(y_f) * 0.1
    
    sigma_f = 0.1 # Used in BNN, not strictly needed for standard MSE, but kept for signature
    
    pde_problem = Poisson1D(x_f, y_f, sigma_f)
    
    # 2. Setup Model and Optimizer
    model = PINN(input_dim=1, output_dim=1, hidden_dims=[20, 20])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 1000
    print(f"Starting standard optimization for {epochs} epochs...")
    
    # 3. Training Loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss using the Baseline PINN method
        total_loss, mse_u, mse_f = model.compute_loss(x_u, y_u, x_f, pde_problem)
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss {total_loss.item():.5f} | MSE_u {mse_u.item():.5f} | MSE_f {mse_f.item():.5f}")
            
    print("Training complete.")
    
    # 4. Plot and Save Results
    print("Generating plot...")
    
    # Define the true solution function for plotting
    def true_u(x):
        return np.sin(np.pi * x)
        
    plot_1d_pinn(
        model=model, 
        x_u=x_u, 
        y_u=y_u, 
        x_f=x_f, 
        y_f=y_f,          # ADDED to show noisy observations on secondary axis
        true_solution_func=true_u,
        title="1D Poisson Equation: PINN Solution (Noisy Data)",
        save_path="experiments/results/poisson_1d_pinn_result.png"
    )

if __name__ == "__main__":
    run_pinn()
