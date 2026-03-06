import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.PINN import PINN
from src.models.BNN import BNN
from src.physics.PDEs import DampedHarmonicOscillator1D
from src.utils.plotting import plot_1d_pinn, plot_1d_bpinn, plot_loss_curves
from src.utils.training import train_pinn, train_bpinn

# =========================================================================
# This script provides a runnable comparison between PINN and B-PINN for 
# the 1D Damped Harmonic Oscillator problem without active forcing.
# m u'' + c u' + k u = 0
# =========================================================================
def warm_start_bnn(pinn, bnn):
    """Transfer weights from trained PINN to BNN, handling the 'net.' prefix mismatch."""
    # Get the state dict from the PINN
    pinn_state_dict = pinn.state_dict()
    
    # Create a new state dict with 'net.' removed from the keys
    new_state_dict = {}
    for key, value in pinn_state_dict.items():
        # If the key starts with 'net.', strip it
        new_key = key.replace('net.', '') 
        new_state_dict[new_key] = value
        
    # Load the cleaned state dict into the BNN
    bnn.load_state_dict(new_state_dict)
    print("BNN successfully initialized with 'net.'-stripped PINN weights.")


def run_damped_oscillator():
    # 1. Setup PDE and Physics Parameters
    m = 1.0
    k = 10.0
    c = 0.5
    sigma_u = 0.01
    sigma_f = 0.01
    
    # 2. Setup Data
    # Time domain from t=0 to t=10
    # Boundary Conditions (Initial Conditions: u(0) = 1)
    # 2. Setup Data
    # Original point: u(0) = 1
    t_0 = torch.tensor([[0.0]], dtype=torch.float32)
    u_0 = torch.tensor([[1.0]], dtype=torch.float32)

    # NEW: Synthetic velocity point: u(0.001) ≈ 1
    # This forces the slope (u') to be near 0 at the start
    t_v = torch.tensor([[0.001]], dtype=torch.float32)
    u_v = torch.tensor([[1.0]], dtype=torch.float32)

    # Combine into boundary tensors
    t_b = torch.cat([t_0, t_v], dim=0)
    u_b = torch.cat([u_0, u_v], dim=0)
    
    # Collocation points (increased density for better physics resolution)
    t_f = torch.linspace(0, 5, 200).view(-1, 1).requires_grad_(True)
    # Target for unforced DHO is 0 everywhere (no explicit forcing)
    u_f_target = torch.zeros_like(t_f)
    
    # Initialize PDE problem
    pde_problem = DampedHarmonicOscillator1D(x_f=t_f, y_f=u_f_target, sigma_f=sigma_f, m=m, c=c, k=k, f=None)
    
    # =========================================================================
    # Standard PINN Baseline
    # =========================================================================
    print("========================================")
    print("Training Standard PINN baseline...")
    # Increased network capacity
    pinn_model = PINN(input_dim=1, output_dim=1, hidden_dims=[30, 30])
    
    pinn_model, history = train_pinn(
        model=pinn_model,
        pde_problem=pde_problem,
        x_b=t_b,
        y_b=u_b,
        x_f=t_f,
        y_f=u_f_target,
        epochs=4000,
        lr=1e-3,
        boundary_weight=10.0
    )

    # =========================================================================
    # Bayesian PINN (HMC)
    # =========================================================================
    print("\n========================================")
    print("Training B-PINN (HMC)...")
    # Reduced from [64,64,64] to prevent HMC dimensionality curse, but wider than baseline
    bnn_model = BNN(input_dim=1, output_dim=1, hidden_dims=[30, 30])
    samples = train_bpinn(
        model=bnn_model,
        pde_problem=pde_problem,
        x_b=t_b,
        y_b=u_b,
        x_f=t_f,
        y_f=u_f_target,
        sigma_u=0.01,  
        sigma_f=0.1,  
        M=100,  # Number of samples to collect
        N=1000, # Total HMC transitions (allowing 900 burn-in)
        L=50,   # Leapfrog steps
        delta_t=0.001, # Finer integration step to prevent massive Energy spikes
        theta_0=None # Enforce random untethered initialization
    )


    
    # =========================================================================
    # Generate Plots
    # =========================================================================
    print("\n========================================")
    print("Generating Plots...")
    
    def true_damped(t):
        if torch.is_tensor(t):
            t = t.detach().cpu().numpy()
        else:
            t = np.array(t)
        omega_n = np.sqrt(k/m)
        zeta = c / (2 * np.sqrt(m * k))
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        A = 1.0
        phi = np.arctan(zeta / np.sqrt(1 - zeta**2))
        return A * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t - phi) / np.cos(phi)

    plot_loss_curves(
        history=history,
        save_path="experiments/results/damped1d_loss.png"
    )

    plot_1d_pinn(
        model=pinn_model,
        x_u=t_b,
        y_u=u_b,
        x_f=t_f,
        y_f=None, 
        true_solution_func=true_damped,
        title="Damped Harmonic Oscillator (PINN)",
        save_path="experiments/results/damped1d_pinn.png"
    )
    
    plot_1d_bpinn(
        model=bnn_model,
        samples=samples,
        x_u=t_b,
        y_u=u_b,
        x_f=t_f,
        y_f=None,
        true_solution_func=true_damped,
        title="Damped Harmonic Oscillator (B-PINN Uncertainty)",
        save_path="experiments/results/damped1d_bpinn.png"
    )
    
if __name__ == "__main__":
    run_damped_oscillator()
