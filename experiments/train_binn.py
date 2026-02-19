import torch
import sys
import os
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.BNN import BNN
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs import Poisson1D

# =========================================================================
# This script provides a runnable entry point to test the B-PINN implementation.
# It sets up a simple 1D Poisson problem (u_xx = f(x)), generates synthetic data, 
# and runs the HMC sampler. 
# It demonstrates passing the required kwargs (x_u, y_u, x_f, y_f, pde_problem) 
# into the sampler, which the sampler proxies to the model.
# =========================================================================

def run_hmc():
    # 1. Setup Data and Physics
    # True function: u(x) = sin(pi * x)
    # PDE: u_xx = -pi^2 * sin(pi * x)
    
    # Boundary data (x_u, y_u)
    x_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y_u = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    
    # Collocation points (x_f)
    x_f = torch.linspace(-1, 1, 20).view(-1, 1).requires_grad_(True)
    y_f = - (torch.pi ** 2) * torch.sin(torch.pi * x_f).detach()
    
    sigma_u = 0.1
    sigma_f = 0.1
    
    pde_problem = Poisson1D(x_f, y_f, sigma_f)
    
    # 2. Setup Model
    model = BNN(input_dim=1, output_dim=1, hidden_dims=[20, 20])
    
    # 3. Setup HMC Parameters
    theta_0 = model.get_weights() # shape (num_params,) - 1D vector
    # Note: here .get_weights() is safe to use as we are not computing any gradients 
    # with respect to this action (no harm in breaking the computational graph)
    M = 50       # Number of samples to keep
    N = 100      # Total HMC iterations
    L = 10       # Leapfrog steps
    delta_t = 0.01 # Step size
    
    print(f"Starting HMC Sampling with {N} total iterations...")
    
    # 4. Run Sampler
    # We pass the physics and data context as **kwargs which are now supported
    samples = HMC_sampler(
        model=model, 
        M=M, 
        N=N, 
        delta_t=delta_t, 
        theta_0=theta_0, 
        L=L,
        # Below are the **kwargs needed by the potential_energy function
        x_u=x_u, 
        y_u=y_u, 
        x_f=x_f, 
        y_f=y_f, 
        sigma_u=sigma_u, 
        sigma_f=sigma_f, 
        pde_problem=pde_problem
    )
    
    print(f"Sampling complete. Gathered {samples.shape[1]} samples of dimension {samples.shape[0]}.")
    
if __name__ == "__main__":
    run_hmc()
