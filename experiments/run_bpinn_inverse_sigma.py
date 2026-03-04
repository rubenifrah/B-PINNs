import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.BNN_Inverse_sigma import BNN_Inverse
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs_inverse import InverseReactionDiffusion1D

# =========================================================================
# Extension of Section 3.3.1: B-PINN-HMC inferring BOTH k AND sigma_u.
#
# PDE:  lambda * u_xx + k * tanh(u) = f,   x in [-0.7, 0.7]
# True solution: u(x) = sin^3(6x)
# lambda = 0.01  (known)
# k = 0.7        (UNKNOWN — inferred)
# sigma_u        (UNKNOWN — inferred)
# sigma_f, sigma_b (KNOWN — fixed as in original paper)
#
# Augmented HMC vector:
#   theta_full = [theta_net, k, log_sigma_u]
#
# Two noise cases, two sigma_u initializations each:
#   Init A: log_sigma_u = log(true_sigma_u)   — near truth
#   Init B: log_sigma_u = log(0.5)            — far from truth
#
# This tests whether sigma_u inference is genuine (data-driven)
# or initialization-dependent, as we found for sigma_f in the forward problem.
# =========================================================================

TRUE_K     = 0.7
LAMBDA_VAL = 0.01


# =========================================================================
# True solution and forcing term
# =========================================================================
def true_u(x):
    return torch.sin(6 * x) ** 3


def true_f(x, lambda_val=LAMBDA_VAL, k=TRUE_K):
    x    = x.clone().detach().requires_grad_(True)
    u    = true_u(x)
    u_x  = torch.autograd.grad(u,   x, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                create_graph=True)[0]
    return (lambda_val * u_xx + k * torch.tanh(u)).detach()


# =========================================================================
# Pretraining
# =========================================================================
def pretrain_network(model, x_u, y_u, x_b, y_b, x_f, y_f,
                     pde_problem, n_steps=3000):
    """
    Pretrain BNN weights using PINN loss before HMC.
    k is fixed at true value during pretraining — only network weights are warmed up.
    sigma_u is not involved in pretraining (it is a noise parameter, not a network weight).
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    k_tensor  = torch.tensor([TRUE_K], dtype=torch.float32)

    for step in range(n_steps):
        optimizer.zero_grad()

        u_pred = model.forward(x_u)
        loss_u = torch.mean((u_pred - y_u) ** 2)

        b_pred = model.forward(x_b)
        loss_b = torch.mean((b_pred - y_b) ** 2)

        x_f_g  = x_f.clone().detach().requires_grad_(True)
        res_f  = pde_problem.compute_residual(model.forward, x_f_g, params=k_tensor)
        loss_f = torch.mean(res_f ** 2)

        loss = loss_u + loss_b + loss_f
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(f"  [Pretrain] step {step:4d} | loss: {loss.item():.6f}")

    print(f"  [Pretrain] Done. Final loss: {loss.item():.6f}\n")


# =========================================================================
# Run one experiment
# =========================================================================
def run_one_experiment(sigma_f, sigma_u_true, sigma_b,
                       log_sigma_u_init, exp_label):
    """
    Runs one full B-PINN-HMC experiment inferring both k and sigma_u.

    Args:
        sigma_f          : true and fixed forcing term noise
        sigma_u_true     : true interior u noise (used for data generation only)
        sigma_b          : true and fixed boundary noise
        log_sigma_u_init : initial log(sigma_u) for HMC
        exp_label        : string label for printing
    """
    torch.manual_seed(42)

    print(f"\n{'=' * 65}")
    print(f"Experiment: {exp_label}")
    print(f"  True sigma_u={sigma_u_true}, init sigma_u={np.exp(log_sigma_u_init):.3f}")
    print(f"  True k={TRUE_K}, init k=0.5")
    print(f"{'=' * 65}")

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------
    x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
    y_b = true_u(x_b) + torch.randn_like(x_b) * sigma_b

    N_u = 20
    N_f = 80

    x_u = torch.linspace(-0.7, 0.7, N_u)[1:-1].view(-1, 1)
    y_u = true_u(x_u) + torch.randn(x_u.shape) * sigma_u_true

    x_f = torch.linspace(-0.7, 0.7, N_f).view(-1, 1).requires_grad_(True)
    y_f = true_f(x_f.detach()) + torch.randn(N_f, 1) * sigma_f

    pde_problem = InverseReactionDiffusion1D(
        x_f, y_f, sigma_f=sigma_f, lambda_val=LAMBDA_VAL
    )

    # ------------------------------------------------------------------
    # Model and pretraining
    # ------------------------------------------------------------------
    model = BNN_Inverse(input_dim=1, output_dim=1, hidden_dims=[50, 50])
    
    # initialise theta
    theta_0 = model.get_initial_theta(log_sigma_u_init=log_sigma_u_init)
    # ------------------------------------------------------------------
    # HMC sampling
    # Note: sigma_u is NOT passed as kwarg — inferred from theta_full
    #       sigma_b and sigma_f ARE passed — fixed and known
    # ------------------------------------------------------------------
    N       = 4000   # was 300
    M       = 1000   # was 100
   
    delta_t = 0.01  # slightly smaller
    L       = 10   # was 20

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
        sigma_b     = sigma_b,
        sigma_f     = sigma_f,
        pde_problem = pde_problem
        # sigma_u intentionally absent — inferred from theta_full
    )

    # ------------------------------------------------------------------
    # Extract posteriors
    # ------------------------------------------------------------------
    k_samples       = model.extract_k_samples(samples)
    sigma_u_samples = model.extract_sigma_u_samples(samples)

    # Chain mixing diagnostic
    mixing = samples[:model.num_params, :].std(dim=1).mean().item()

    print(f"Chain mixing (mean std across weights): {mixing:.5f}")
    print(f"k        | mean: {k_samples.mean().item():.4f}  "
          f"std: {k_samples.std().item():.2e}  (true: {TRUE_K})")
    print(f"sigma_u  | mean: {sigma_u_samples.mean().item():.4f}  "
          f"std: {sigma_u_samples.std().item():.2e}  (true: {sigma_u_true})")

    data = dict(x_u=x_u, y_u=y_u, x_b=x_b, y_b=y_b,
                x_f=x_f, y_f=y_f, sigma_u_true=sigma_u_true)

    return samples, model, data, k_samples, sigma_u_samples, pde_problem


# =========================================================================
# Plotting
# =========================================================================
def plot_results(all_results, save_path):
    """
    Produces a figure with one row per experiment showing:
        Col 1: posterior predictive u(x)
        Col 2: posterior histogram over k
        Col 3: posterior histogram over sigma_u
        Col 4: sigma_u trace plot
    """
    x_test    = torch.linspace(-0.7, 0.7, 200).view(-1, 1)
    x_np      = x_test.numpy().flatten()
    u_true_np = true_u(x_test).numpy().flatten()

    n_exp = len(all_results)
    fig, axes = plt.subplots(n_exp, 4, figsize=(20, 5 * n_exp))
    if n_exp == 1:
        axes = axes[np.newaxis, :]

    for row, (label, result) in enumerate(all_results.items()):
        samples, model, data, k_samp, sigma_u_samp, pde_problem = result
        sigma_u_true = data['sigma_u_true']

        # Posterior predictive u(x)
        u_preds = []
        for i in range(samples.shape[1]):
            theta_net = samples[:model.num_params, i]
            with torch.no_grad():
                u_preds.append(model.functional_forward(theta_net, x_test).numpy())
        u_preds = np.array(u_preds).squeeze()
        u_mean  = u_preds.mean(0)
        u_std   = u_preds.std(0)

        x_train = torch.cat([data['x_u'], data['x_b']]).numpy().flatten()
        y_train = torch.cat([data['y_u'], data['y_b']]).numpy().flatten()

        # Col 1: u(x) posterior predictive
        ax = axes[row, 0]
        ax.fill_between(x_np, u_mean - 2*u_std, u_mean + 2*u_std,
                        alpha=0.4, color='cyan', label='±2 std')
        ax.plot(x_np, u_true_np, 'k-',  linewidth=2,   label='Exact')
        ax.plot(x_np, u_mean,    'r--', linewidth=1.5, label='Mean')
        ax.scatter(x_train, y_train, c='none', edgecolors='blue',
                   s=30, zorder=5, label='Training data')
        ax.set_title(f"{label}\nu(x) posterior")
        ax.set_xlabel("x"); ax.set_ylabel("u")
        ax.set_xlim(-0.7, 0.7); ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Col 2: k posterior histogram
        ax = axes[row, 1]
        ax.hist(k_samp.numpy(), bins=30, alpha=0.7, color='blue',
                density=True, label='Posterior k')
        ax.axvline(TRUE_K, color='black', linestyle='--', linewidth=2,
                   label=f'True k={TRUE_K}')
        ax.axvline(k_samp.mean().item(), color='red', linestyle='-', linewidth=2,
                   label=f'Mean={k_samp.mean().item():.3f}')
        ax.set_title("Posterior over k")
        ax.set_xlabel("k"); ax.set_ylabel("Density")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Col 3: sigma_u posterior histogram
        ax = axes[row, 2]
        ax.hist(sigma_u_samp.numpy(), bins=30, alpha=0.7, color='orange',
                density=True, label='Posterior σ_u')
        ax.axvline(sigma_u_true, color='black', linestyle='--', linewidth=2,
                   label=f'True σ_u={sigma_u_true}')
        ax.axvline(sigma_u_samp.mean().item(), color='red', linestyle='-',
                   linewidth=2,
                   label=f'Mean={sigma_u_samp.mean().item():.4f}')
        ax.set_title("Posterior over σ_u")
        ax.set_xlabel("σ_u"); ax.set_ylabel("Density")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Col 4: sigma_u trace
        ax = axes[row, 3]
        log_sigma_u_trace = samples[model.idx_log_sigma_u, :].numpy()
        ax.plot(log_sigma_u_trace, color='orange', linewidth=1)
        ax.axhline(np.log(sigma_u_true), color='black', linestyle='--',
                   linewidth=2, label=f'True log(σ_u)={np.log(sigma_u_true):.3f}')
        ax.set_title("log(σ_u) trace")
        ax.set_xlabel("HMC sample index"); ax.set_ylabel("log(σ_u)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

    plt.suptitle("B-PINN Inverse Problem: Inferring k and σ_u jointly",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nFigure saved to {save_path}")
    plt.show()


# =========================================================================
# Main
# =========================================================================
def run():
    all_results = {}

    # ------------------------------------------------------------------
    # Noise case 1: sigma = 0.01
    # Two initializations: near truth and far from truth
    # ------------------------------------------------------------------
    all_results["Case 1 | σ=0.01 | init σ_u=0.01 (near truth)"] = run_one_experiment(
        sigma_f          = 0.01,
        sigma_u_true     = 0.01,
        sigma_b          = 0.01,
        log_sigma_u_init = np.log(0.01),   # near truth
        exp_label        = "Case 1 | sigma=0.01 | init near truth"
    )

    all_results["Case 1 | σ=0.01 | init σ_u=0.5 (far from truth)"] = run_one_experiment(
        sigma_f          = 0.01,
        sigma_u_true     = 0.01,
        sigma_b          = 0.01,
        log_sigma_u_init = np.log(0.5),    # far from truth
        exp_label        = "Case 1 | sigma=0.01 | init far from truth"
    )

    # ------------------------------------------------------------------
    # Noise case 2: sigma_u = 0.1
    # Two initializations
    # ------------------------------------------------------------------
    all_results["Case 2 | σ_u=0.1 | init σ_u=0.1 (near truth)"] = run_one_experiment(
        sigma_f          = 0.1,
        sigma_u_true     = 0.1,
        sigma_b          = 0.01,
        log_sigma_u_init = np.log(0.1),    # near truth
        exp_label        = "Case 2 | sigma_u=0.1 | init near truth"
    )

    all_results["Case 2 | σ_u=0.1 | init σ_u=0.5 (far from truth)"] = run_one_experiment(
        sigma_f          = 0.1,
        sigma_u_true     = 0.1,
        sigma_b          = 0.01,
        log_sigma_u_init = np.log(0.5),    # far from truth
        exp_label        = "Case 2 | sigma_u=0.1 | init far from truth"
    )

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Summary: Inferred k and sigma_u across all experiments")
    print(f"{'Experiment':<45} {'k mean':>8} {'k std':>10} "
          f"{'σ_u mean':>10} {'σ_u std':>10}")
    print("-" * 70)
    for label, result in all_results.items():
        _, _, data, k_samp, sigma_u_samp, _ = result
        print(f"{label[:44]:<45} "
              f"{k_samp.mean().item():>8.3f} "
              f"{k_samp.std().item():>10.2e} "
              f"{sigma_u_samp.mean().item():>10.4f} "
              f"{sigma_u_samp.std().item():>10.2e}")
    print(f"{'True values':<45} {TRUE_K:>8.3f} {'—':>10}")
    print("=" * 70)

    plot_results(
        all_results,
        save_path="experiments/results/bpinn_inverse_sigma_u.png"
    )


if __name__ == "__main__":
    run()