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

def run_damped_oscillator():
    # 1. Setup PDE and Physics Parameters
    m = 1.0
    k = 10.0
    c = 0.5
    
    # 2. Setup Data
    # Time domain from t=0 to t=10
    # Boundary Conditions (Initial Conditions: u(0) = 1)
    t_b = torch.tensor([[0.0]], dtype=torch.float32)
    u_b = torch.tensor([[1.0]], dtype=torch.float32)
    
    # Collocation points
    t_f = torch.linspace(0, 5, 40).view(-1, 1).requires_grad_(True)
    # Target for unforced DHO is 0 everywhere (no explicit forcing)
    u_f_target = torch.zeros_like(t_f)
    
    # Initialize PDE problem
    pde_problem = DampedHarmonicOscillator1D(x_f=t_f, y_f=u_f_target, sigma_f=0.1, m=m, c=c, k=k, f=None)
    
    # =========================================================================
    # Standard PINN Baseline
    # =========================================================================
    print("========================================")
    print("Training Standard PINN baseline...")
    pinn_model = PINN(input_dim=1, output_dim=1, hidden_dims=[30, 30])
    
    pinn_model, history = train_pinn(
        model=pinn_model,
        pde_problem=pde_problem,
        x_b=t_b,
        y_b=u_b,
        x_f=t_f,
        y_f=u_f_target,
        epochs=1500,
        lr=2e-3,
        boundary_weight=100.0
    )

    # =========================================================================
    # Bayesian PINN (HMC)
    # =========================================================================
    print("\n========================================")
    print("Training B-PINN (HMC)...")
    bnn_model = BNN(input_dim=1, output_dim=1, hidden_dims=[30, 30])
    
    samples = train_bpinn(
        model=bnn_model,
        pde_problem=pde_problem,
        x_b=t_b,
        y_b=u_b,
        x_f=t_f,
        y_f=u_f_target,
        sigma_u=0.01,  # Tight variance enforces initial condition heavily to prevent collapse
        sigma_f=0.1,  
        M=50,
        N=150,
        L=15,
        delta_t=0.001
    )
    
    # =========================================================================
    # Generate Plots
    # =========================================================================
    print("\n========================================")
    print("Generating Plots...")
    
    def true_damped(t):
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
