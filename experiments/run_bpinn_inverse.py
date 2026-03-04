import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.BNN_Inverse import BNN_Inverse
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs_inverse import InverseReactionDiffusion1D

# =========================================================================
# Reproduces Section 3.3.1 of Yang et al. (2020) — B-PINN-HMC only.
#
# PDE:  lambda * u_xx + k * tanh(u) = f,   x in [-0.7, 0.7]
# True solution: u(x) = sin^3(6x)
# lambda = 0.01  (known)
# k = 0.7        (UNKNOWN — inferred via HMC)
#
# Dataset:
#   D_f : 32 equidistant sensors for f
#   D_u : 6 interior sensors for u
#   D_b : 2 boundary sensors at x = +-0.7
#
# Two noise cases tested (Table 1 / Figure 7 of the paper):
#   Case 1: epsilon_f ~ N(0, 0.01^2), epsilon_u ~ N(0, 0.01^2), epsilon_b ~ N(0, 0.01^2)
#   Case 2: epsilon_f ~ N(0, 0.1^2),  epsilon_u ~ N(0, 0.1^2),  epsilon_b ~ N(0, 0.01^2)
#
# Output:
#   - Figure 7 style plots: u(x) and f(x) for both noise cases
#   - Table 1 style report: mean and std of inferred k
# =========================================================================

TRUE_K     = 0.7
LAMBDA_VAL = 0.01


# =========================================================================
# True solution and forcing term
# =========================================================================
def true_u(x):
    """u(x) = sin^3(6x)"""
    return torch.sin(6 * x) ** 3


def true_f(x, lambda_val=LAMBDA_VAL, k=TRUE_K):
    """
    f(x) derived analytically from the PDE: lambda*u_xx + k*tanh(u) = f
    Uses autograd on the true solution.
    """
    x   = x.clone().detach().requires_grad_(True)
    u   = true_u(x)
    u_x = torch.autograd.grad(u,   x,   grad_outputs=torch.ones_like(u),
                               create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x,   grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    f = lambda_val * u_xx + k * torch.tanh(u)
    return f.detach()



# =========================================================================
# Run one noise case
# =========================================================================
def run_one_case(sigma_f, sigma_u, sigma_b, case_label):
    """
    Runs the full B-PINN-HMC experiment for one noise configuration.

    Returns:
        samples      : HMC posterior samples, shape (total_params, M)
        model        : trained BNN_Inverse instance
        data         : dict of all data tensors (for plotting)
        k_samples    : posterior samples of k, shape (M,)
    """
    torch.manual_seed(42)

    print("=" * 60)
    print(f"Case {case_label}: sigma_f={sigma_f}, sigma_u={sigma_u}, sigma_b={sigma_b}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data generation (Section 3.3.1)
    # ------------------------------------------------------------------
    # D_b: 2 boundary sensors at x = -0.7 and x = 0.7
  
    x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
    y_b = true_u(x_b) + torch.randn_like(x_b) * sigma_b

    N_u = 6
    # D_u: 6 interior sensors uniformly placed in (-0.7, 0.7)
    x_u = torch.linspace(-0.7, 0.7, N_u)[1:-1].view(-1, 1)  # 6 points
    y_u = true_u(x_u) + torch.randn(x_u.shape) * sigma_u

    N_f = 32
    x_f = torch.linspace(-0.7, 0.7, N_f).view(-1, 1).requires_grad_(True)
    y_f = true_f(x_f.detach())
    y_f = y_f + torch.randn_like(y_f) * sigma_f

    pde_problem = InverseReactionDiffusion1D(
        x_f, y_f, sigma_f=sigma_f, lambda_val=LAMBDA_VAL
    )

    # ------------------------------------------------------------------
    # Model and pretraining
    # ------------------------------------------------------------------
    model = BNN_Inverse(input_dim=1, output_dim=1, hidden_dims=[50, 50])
    print(f"Network params: {model.num_params}, Total HMC dim: {model.total_params}\n")

    print("Step 1: Pretraining...")

    # ------------------------------------------------------------------
    # HMC sampling
    # ------------------------------------------------------------------
    K_INIT  = 0.5   # deliberately away from true 0.7
    theta_0 = model.get_initial_theta(k_init=K_INIT)

    print(f"Step 2: HMC sampling (init k={K_INIT}, true k={TRUE_K})...")

    M       = 2000   # Increased from 500
    N       = 15000   # Increased from 2000 (Burn-in)
    L       = 50     # More leapfrog steps can help explore further per iteration
    delta_t = 0.1 # Smaller step size to avoid immediate divergence

    samples = HMC_sampler(
        model   = model,
        M       = M,
        N       = N,
        delta_t = delta_t,
        theta_0 = theta_0,
        L       = L,
        x_u         = x_u,
        y_u         = y_u,
        x_b         = x_b,
        y_b         = y_b,
        x_f         = x_f,
        y_f         = y_f,
        sigma_u     = sigma_u,
        sigma_b     = sigma_b,
        sigma_f     = sigma_f,
        pde_problem = pde_problem
    )

    k_samples = model.extract_k_samples(samples)

    print(f"k | mean: {k_samples.mean().item():.4f}  "
          f"std: {k_samples.std().item():.2e}  "
          f"(true: {TRUE_K})\n")

    data = dict(x_u=x_u, y_u=y_u, x_b=x_b, y_b=y_b, x_f=x_f, y_f=y_f)

    return samples, model, data, k_samples, pde_problem


# =========================================================================
# Posterior predictive
# =========================================================================
def compute_posterior_predictive(model, samples, x_test, pde_problem):
    """
    Computes posterior predictive mean and std for u(x) and f(x)
    over all M posterior samples.

    Returns:
        u_mean, u_std : shape (N_test,)
        f_mean, f_std : shape (N_test,)
    """
    u_preds = []
    f_preds = []

    for i in range(samples.shape[1]):
        theta_net = samples[:model.num_params, i]
        k_val     = samples[model.num_params, i]

        with torch.no_grad():
            u_pred = model.functional_forward(theta_net, x_test)
        u_preds.append(u_pred.numpy())

        f_pred = model.predict_f(theta_net, x_test, k_val, LAMBDA_VAL)
        f_preds.append(f_pred.numpy())

    u_preds = np.array(u_preds).squeeze()
    f_preds = np.array(f_preds).squeeze()

    return (u_preds.mean(0), u_preds.std(0),
            f_preds.mean(0), f_preds.std(0))


# =========================================================================
# Plotting — reproduces Figure 7 of the paper
# =========================================================================
def plot_figure7(results, save_path):
    """
    Reproduces Figure 7 of Yang et al. (2020) for B-PINN-HMC.

    Layout: 2 rows (one per noise case), each row has 2 subplots:
        - Left:  posterior predictive for u(x)
        - Right: posterior predictive for f(x)
    """
    x_test   = torch.linspace(-0.7, 0.7, 200).view(-1, 1)
    x_np     = x_test.numpy().flatten()
    u_true_np = true_u(x_test).numpy().flatten()
    f_true_np = true_f(x_test).numpy().flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    case_labels = ['(a) Noise scale 0.01', '(b) Noise scale 0.1']

    for row, (case_label, (samples, model, data, k_samples, pde_problem)) in \
            enumerate(zip(case_labels, results)):

        u_mean, u_std, f_mean, f_std = compute_posterior_predictive(
            model, samples, x_test, pde_problem
        )

        # Combine all training observations of u for scatter plot
        x_train_u = torch.cat([data['x_u'], data['x_b']], dim=0).numpy().flatten()
        y_train_u = torch.cat([data['y_u'], data['y_b']], dim=0).numpy().flatten()

        x_train_f = data['x_f'].detach().numpy().flatten()
        y_train_f = data['y_f'].numpy().flatten()

        # --- Left subplot: u(x) ---
        ax = axes[row, 0]
        ax.fill_between(x_np,
                        u_mean - 2 * u_std,
                        u_mean + 2 * u_std,
                        alpha=0.4, color='cyan', label='2 std')
        ax.plot(x_np, u_true_np, 'k-',  linewidth=2,   label='Exact')
        ax.plot(x_np, u_mean,    'r--', linewidth=1.5, label='Mean')
        ax.scatter(x_train_u, y_train_u,
                   c='none', edgecolors='blue', s=40, zorder=5, label='Training data')
        ax.set_title(f"B-PINN-HMC  {case_label}\n"
                     f"k: {k_samples.mean().item():.3f} ± {k_samples.std().item():.2e}  "
                     f"(true: {TRUE_K})")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-3, 2)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

        # --- Right subplot: f(x) ---
        ax = axes[row, 1]
        ax.fill_between(x_np,
                        f_mean - 2 * f_std,
                        f_mean + 2 * f_std,
                        alpha=0.4, color='cyan', label='2 std')
        ax.plot(x_np, f_true_np, 'k-',  linewidth=2,   label='Exact')
        ax.plot(x_np, f_mean,    'r--', linewidth=1.5, label='Mean')
        ax.scatter(x_train_f, y_train_f,
                   c='none', edgecolors='blue', s=40, zorder=5, label='Training data')
        ax.set_title(f"Forcing term f(x)  {case_label}")
        ax.set_xlabel("x")
        ax.set_ylabel("f")
        ax.set_xlim(-0.7, 0.7)
        ax.set_ylim(-2, 2)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle("Figure 7 (reproduced): 1D Diffusion-Reaction Inverse Problem — B-PINN-HMC",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Figure saved to {save_path}")
    plt.show()


# =========================================================================
# Table 1 report
# =========================================================================
def print_table1(results, case_labels):
    """
    Prints Table 1 style report for inferred k.
    """
    print("\n" + "=" * 50)
    print("Table 1 (B-PINN-HMC only): Inferred k")
    print(f"{'Noise scale':<15} {'Mean':>10} {'Std':>15}")
    print("-" * 50)
    for label, (samples, model, data, k_samples, _) in zip(case_labels, results):
        mean = k_samples.mean().item()
        std  = k_samples.std().item()
        print(f"{label:<15} {mean:>10.3f} {std:>15.2e}")
    print(f"{'True k':<15} {TRUE_K:>10.3f}")
    print("=" * 50 + "\n")


# =========================================================================
# Main
# =========================================================================
def run():
    # Case 1: all sigma = 0.01  (Figure 7a / Table 1 row 1)
    results_case1 = run_one_case(
        sigma_f=0.01, sigma_u=0.01, sigma_b=0.01, case_label=1
    )

    # Case 2: sigma_f=0.1, sigma_u=0.1, sigma_b=0.01  (Figure 7b / Table 1 row 2)
    results_case2 = run_one_case(
        sigma_f=0.1, sigma_u=0.1, sigma_b=0.01, case_label=2
    )

    results      = [results_case1, results_case2]
    case_labels  = ["0.01", "0.1"]

    print_table1(results, case_labels)

    plot_figure7(
        results,
        save_path="experiments/results/figure7_bpinn_hmc_inverse.png"
    )


if __name__ == "__main__":
    run()