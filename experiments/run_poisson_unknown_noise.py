import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.BNN_unknown_noise import BNN_UnknownNoise
from src.samplers.HMC import HMC_sampler
from src.models.PINN import PINN
from src.physics.PDEs import Poisson1D
import torch.optim as optim
from src.utils.metrics import evaluate_uncertainty, compute_ece_and_plot, compute_relative_l2

# =========================================================================
# Extension experiment: B-PINN with unknown noise levels.
#
# The true noise is sigma_u = sigma_f = 0.1, but we do NOT tell the model this.
# Instead, log_sigma_u and log_sigma_f are sampled jointly with the BNN weights.
# After sampling, we check whether the inferred sigma posteriors recover the
# true values — this is the main empirical question of the extension.
#
# Compare against train_bpinn.py where sigma is fixed and known.
# =========================================================================

TRUE_SIGMA_U = 0.01
TRUE_SIGMA_F = 0.01

def run_hmc_unknown_noise():
    lambd = 0.01  # Diffusion coefficient from paper

    # ------------------------------------------------------------------
    # 1. Generate data — identical setup to train_bpinn.py
    # ------------------------------------------------------------------
    torch.manual_seed(42)

    # 1. Setup Data and Physics
    # Boundary data (x_b, y_b)
    x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
    y_b = torch.sin(6 * x_b)**3
    y_b = y_b + torch.randn_like(y_b) * TRUE_SIGMA_U  # add noise

    # Collocation points (x_f), forcing term measurements (y_f)
    Nbr_colloc = 80
    x_f = torch.linspace(-0.7, 0.7, Nbr_colloc).view(-1, 1).requires_grad_(True)
    y_f = lambd * (216 * torch.sin(6 * x_f) * torch.cos(6 * x_f)**2 - 108 * torch.sin(6 * x_f)**3).detach()
    y_f = y_f + torch.randn_like(y_f) * TRUE_SIGMA_F

    pde_problem = Poisson1D(x_f, y_f, sigma_f=None, lambd=lambd)
    
    def true_u(x):
        return np.sin(6 * x)**3    

    # ------------------------------------------------------------------
    # 2. Setup model
    # ------------------------------------------------------------------
    # Set the mean of the log-sigma prior
    mu_log_sigma = -2
    model = BNN_UnknownNoise(input_dim=1, output_dim=1, hidden_dims=[50, 50], mu_log_sigma=mu_log_sigma)

    print(f"Network parameters:  {model.num_params}")
    print(f"Total HMC dimension: {model.total_params}  (+ log_sigma_u, log_sigma_f)")

    # Pretrain first
    
    # Then initialize theta_0 
    theta_0 = model.get_initial_theta(log_sigma_f_init=-2)
  
    print(f"\nInitial log_sigma_f: {theta_0[model.num_params].item():.3f}  "
          f"=> sigma_f = {torch.exp(theta_0[model.num_params]).item():.3f}")
    print(f"Fixed sigma_u = {TRUE_SIGMA_U}, True sigma_f = {TRUE_SIGMA_F}\n")

    # ------------------------------------------------------------------
    # 4. HMC parameters
    # NOTE: sigma_u and sigma_f are NO LONGER passed as kwargs —
    # they are inferred from theta_full inside potential_energy().
    # ------------------------------------------------------------------
    N = 7000   
    M = 1000   
    delta_t = 0.01  
    L = 10   
    
    print(f"Starting HMC with {N} iterations, keeping last {M} samples...")

    samples = HMC_sampler(
        model=model,
        M=M,
        N=N,
        delta_t=delta_t,
        theta_0=theta_0,
        L=L,
        # kwargs passed to potential_energy — note: no sigma_u / sigma_f here
        x_u=x_b,
        y_u=y_b,
        x_f=x_f,
        y_f=y_f,
        pde_problem=pde_problem,
        sigma_u = TRUE_SIGMA_U
    )

    # Check acceptance rate proxy — if std is near zero, chain is stuck
    theta_net_samples = samples[:model.num_params, :]
    print(f"Chain mixing check - mean std across weights: "
        f"{theta_net_samples.std(dim=1).mean().item():.5f}")

    # Also plot the sigma trace to see if it moved
    plt.figure()
    # log_sigma_f is now at index model.num_params
    plt.plot(samples[model.num_params, :].numpy(), label='log_sigma_f trace', color='orange')
    plt.legend()
    plt.title("Sigma_f trace — should look like noise, not a flat line")
    plt.savefig("experiments/results/sigma_trace.png")

    print(f"Sampling complete. Samples shape: {samples.shape}")

    # ------------------------------------------------------------------
    # 5. Extract and report inferred noise levels
    # ------------------------------------------------------------------
    sigma_f_samples = model.extract_sigma_samples(samples)

    print("\n--- Inferred Noise Levels ---")
    print(f"sigma_f | mean: {sigma_f_samples.mean().item():.4f}  "
          f"std: {sigma_f_samples.std().item():.4f}  "
          f"(true: {TRUE_SIGMA_F})")

    # =========================================================================
    # Evaluate Uncertainty Metrics (PICP & MPIW & NLL)
    # =========================================================================
    print("\n========================================")
    print("Evaluating B-PINN Uncertainty Metrics...")
    
    x_test = torch.linspace(-0.7, 0.7, 200).view(-1, 1)
    y_true_test = torch.tensor(true_u(x_test.numpy()), dtype=torch.float32)
    theta_net_samples = samples[:model.num_params, :]
    picp, mpiw, nll, l2 = evaluate_uncertainty(model, theta_net_samples, x_test, y_true_test, n_std=2.0)


    print(f"Target Coverage: ~95.4% (using 2-sigma bounds)")
    print(f"PICP: {picp * 100:.2f}% of the true solution is captured within bounds.")
    print(f"MPIW: {mpiw:.4f} average width of the uncertainty interval.")
    print(f"Mean NLL: {nll:.4f} (Lower is better)")
    print(f"L2 relative error: {l2:.4f}")
    
    # Passing model (not bnn_model) and theta_net_samples
    ece = compute_ece_and_plot(
        model=model, 
        samples=theta_net_samples, 
        x_test=x_test, 
        y_true=y_true_test, 
        num_bins=15, 
        save_path="experiments/results/poisson_1d_reliability.png"
    )
    print(f"Expected Calibration Error (ECE): {ece:.4f}")

    # ------------------------------------------------------------------
    # 6. Plot results
    # ------------------------------------------------------------------
    x_test = torch.linspace(-0.7, 0.7, 200).view(-1, 1)
    u_true = true_u(x_test)

    # Collect posterior predictive samples
    u_preds = []
    for i in range(samples.shape[1]):
        theta_net = samples[:model.num_params, i]
        with torch.no_grad():
            u_pred = model.functional_forward(theta_net, x_test)
        u_preds.append(u_pred.numpy())

    u_preds = np.array(u_preds).squeeze()  # (M, N_test)
    u_mean = u_preds.mean(axis=0)
    u_std  = u_preds.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PDE solution with uncertainty
    ax = axes[0]
    x_np = x_test.numpy().flatten()
    ax.plot(x_np, u_true, 'k-', label='True u(x)', linewidth=2)
    ax.plot(x_np, u_mean, 'r--', label='Posterior mean', linewidth=2)
    ax.fill_between(x_np, u_mean - 2*u_std, u_mean + 2*u_std,
                    alpha=0.3, color='cyan', label='±2 std')
    ax.scatter(x_b.numpy(), y_b.numpy(), c='blue', zorder=5, label='Noisy observations')
    ax.set_title("B-PINN Solution (Unknown Noise)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Posterior distribution over sigma_f
    ax = axes[1]
    ax.hist(sigma_f_samples.numpy(), bins=20, alpha=0.6,
            color='orange', label=f'Inferred σ_f', density=True)
    ax.axvline(TRUE_SIGMA_F, color='orange', linestyle='--', linewidth=2, label=f'True σ_f ({TRUE_SIGMA_F})')
    # We can just plot a blue line for the fixed sigma_u so it's still on the graph
    ax.axvline(TRUE_SIGMA_U, color='blue', linestyle='-', linewidth=2, label=f'Fixed σ_u ({TRUE_SIGMA_U})')
    
    ax.set_title("Posterior over Noise Levels")
    ax.set_xlabel("σ")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs("experiments/results", exist_ok=True)
    save_path = "experiments/results/bpinn_unknown_noise.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.show()


if __name__ == "__main__":
    run_hmc_unknown_noise()