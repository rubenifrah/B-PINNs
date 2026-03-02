import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.PINN import PINN
from src.models.BNN import BNN
from src.physics.PDEs import Poisson1D
from src.utils.plotting import plot_1d_pinn, plot_1d_bpinn, plot_loss_curves
from src.utils.training import train_pinn, train_bpinn
from src.utils.metrics import evaluate_uncertainty, compute_ece_and_plot, compute_relative_l2

# =========================================================================
# This script provides a runnable comparison between PINN and B-PINN for 
# the 1D Poisson Equation (u_xx = f(x)) with noisy synthetic data.
# =========================================================================

def run_poisson_experiment():
    lambd = 0.01 # Diffusion coefficient from paper

    # 1. Setup Data and Physics
    # Boundary data (x_b, y_b)
    x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
    y_b = torch.sin(6 * x_b)**3

   
    # Collocation points (x_f), forcing term measurements (y_f)
    Nbr_colloc= 80
    x_f = torch.linspace(-0.7, 0.7, Nbr_colloc).view(-1, 1).requires_grad_(True)
    y_f = lambd * (216 * torch.sin(6 * x_f) * torch.cos(6 * x_f)**2 - 108 * torch.sin(6 * x_f)**3).detach()
    # Add noisy data mirroring the standard testing setup
    noise_std = 0.01
    y_b = y_b + torch.randn_like(y_b) * noise_std
    y_f = y_f + torch.randn_like(y_f) * noise_std
   
    
    sigma_u = noise_std
    sigma_f = noise_std
    
    pde_problem = Poisson1D(x_f, y_f, sigma_f, lambd=lambd)    

    def true_u(x):
        return np.sin(6 * x)**3
    # =========================================================================
    # Standard PINN Baseline
    # =========================================================================
    print("========================================")
    print("Training Standard PINN baseline...")
    pinn_model = PINN(input_dim=1, output_dim=1, hidden_dims=[50, 50])
    
    pinn_model, history = train_pinn(
        model=pinn_model,
        pde_problem=pde_problem,
        x_b=x_b,
        y_b=y_b,
        x_f=x_f,
        y_f=y_f,
        epochs=2500,
        lr=1e-3
    )

    pinn_model.eval()
    with torch.no_grad():
        # Predict on the same x_test used for BNN
        x_test = torch.linspace(-0.7, 0.7, 500).view(-1, 1)
        y_pred_pinn = pinn_model(x_test)
        y_true_test = torch.tensor(true_u(x_test.numpy()), dtype=torch.float32)
        
        l2_pinn = compute_relative_l2(y_pred_pinn, y_true_test)
    
    print(f"Standard PINN Relative L2 Error: {l2_pinn:.4f}")

    # =========================================================================
    # Bayesian PINN (HMC)
    # =========================================================================
    print("\n========================================")
    print("Training B-PINN (HMC)...")
    bnn_model = BNN(input_dim=1, output_dim=1, hidden_dims=[50, 50])
    samples = train_bpinn(
        model=bnn_model,
        pde_problem=pde_problem,
        x_b=x_b,
        y_b=y_b,
        x_f=x_f,
        y_f=y_f,
        sigma_u=sigma_u,
        sigma_f=sigma_f,
        M=2000,       
        N=1000,      
        L=10,       
        delta_t=0.01
    )

        

    # =========================================================================
    # Evaluate Uncertainty Metrics (PICP & MPIW & NLL)
    # =========================================================================
    print("\n========================================")
    print("Evaluating B-PINN Uncertainty Metrics...")
    
  
    
    # Calculate using 2 standard deviations 
    picp, mpiw, nll,l2 = evaluate_uncertainty(bnn_model, samples, x_test, y_true_test, n_std=2.0)
    
    print(f"Target Coverage: ~95.4% (using 2-sigma bounds)")
    print(f"PICP: {picp * 100:.2f}% of the true solution is captured within bounds.")
    print(f"MPIW: {mpiw:.4f} average width of the uncertainty interval.")
    print(f"Mean NLL: {nll:.4f} (Lower is better)")
    print(f"L2 relative error: {l2:.4f} ")


    ece = compute_ece_and_plot(
        model=bnn_model, 
        samples=samples, 
        x_test=x_test, 
        y_true=y_true_test, 
        num_bins=15, 
        save_path="experiments/results/poisson_1d_reliability.png"
    )
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # =========================================================================
    # Generate Plots
    # =========================================================================
    print("\n========================================")
    print("Generating plots...")
    

    plot_loss_curves(
        history=history,
        save_path="experiments/results/poisson_1d_loss.png"
    )

    plot_1d_pinn(
        model=pinn_model, 
        x_u=x_b, 
        y_u=y_b, 
        x_f=x_f, 
        y_f=y_f,
        true_solution_func=true_u,
        title="1D Poisson Equation: PINN Solution (Noisy Data)",
        save_path="experiments/results/poisson_1d_pinn_result.png"
    )

    plot_1d_bpinn(
        model=bnn_model, 
        samples=samples,
        x_u=x_b, 
        y_u=y_b, 
        x_f=x_f, 
        y_f=y_f,
        true_solution_func=true_u,
        title="1D Poisson Equation: B-PINN Solution (Noisy Data)",
        save_path="experiments/results/poisson_1d_bpinn_result.png"
    )

if __name__ == "__main__":
    run_poisson_experiment()
