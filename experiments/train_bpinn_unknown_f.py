import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.BNN_unknown_f import BNN_UnknownNoise_f
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs import Poisson1D

# =========================================================================
# Extension experiment: B-PINN with unknown forcing term noise sigma_f.
#
# This script mirrors the setup of train_bpinn.py (1D Poisson, same data)
# but removes the assumption that sigma_f is known. Instead, sigma_f is
# inferred jointly with the BNN weights via HMC.
#
# sigma_b (boundary noise) is kept fixed and known, following the original
# paper's treatment. This is justified because N_b = 2 boundary points are
# insufficient to reliably infer sigma_b from data — the posterior would be
# dominated by the prior rather than the observations.
#
# Key differences vs train_bpinn.py:
#   - BNN_UnknownNoise replaces BNN (theta_full has one extra dimension)
#   - sigma_f is NOT passed to HMC_sampler — it is inferred internally
#   - sigma_b IS passed as a fixed known value, as in the original paper
#   - Network weights are pretrained via PINN loss before HMC (required
#     because random initialization in 482 dimensions causes total rejection)
# =========================================================================

# Ground truth noise levels (used for data generation and comparison)
TRUE_SIGMA_B = 0.1   # boundary noise  — kept fixed and known
TRUE_SIGMA_F = 0.1   # forcing noise   — inferred by our extension


# =========================================================================
# Step 1: Pretraining
# =========================================================================
def pretrain_network(model, x_b, y_b, x_f, y_f, pde_problem, n_steps=2000):
    """
    Pretrain BNN weights using standard PINN loss (Adam optimizer) before HMC.

    Why this is necessary:
        HMC explores the posterior by computing gradients of U(theta_full).
        Starting from random weights, the PDE residual is huge, making U very
        large and any proposed step equally bad — the Metropolis step rejects
        every proposal, the chain never moves (flat trace, std=0).

        Pretraining brings the weights to a region where the PDE is approximately
        satisfied. HMC then only explores a small neighborhood around a good
        solution, where energy differences between proposals are manageable.

    Note: pretraining uses the standard forward() pass and Adam, not HMC.
          It does not involve sigma_f — it simply minimizes PDE and data residuals.

    Args:
        model      : BNN_UnknownNoise instance (standard forward pass used here)
        x_b, y_b   : boundary locations and noisy observations
        x_f, y_f   : collocation points and noisy forcing term measurements
        pde_problem: PDE residual evaluator
        n_steps    : number of Adam steps (2000 is sufficient for this problem)
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for step in range(n_steps):
        optimizer.zero_grad()

        # Boundary condition loss: ũ(x_b) ≈ b̄
        b_pred = model.forward(x_b)
        loss_b = torch.mean((b_pred - y_b) ** 2)

        # PDE residual loss: N_x(ũ; lambda) ≈ f̄
        x_f_grad = x_f.clone().detach().requires_grad_(True)
        res_f    = pde_problem.compute_residual(model.forward, x_f_grad)
        loss_f   = torch.mean(res_f ** 2)

        loss = loss_b + loss_f
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"  [Pretrain] step {step:4d} | loss: {loss.item():.6f} "
                  f"(boundary: {loss_b.item():.6f}, physics: {loss_f.item():.6f})")

    print(f"  [Pretrain] Done. Final loss: {loss.item():.6f}\n")

def true_forcing(x):
    # f(x) for u(x)=sin(pi x): u_xx = -pi^2 sin(pi x)
    return -(torch.pi ** 2) * torch.sin(torch.pi * x)

def make_forcing_data(n, sigma, device=None):
    x = torch.linspace(-1, 1, n).view(-1, 1)
    if device is not None:
        x = x.to(device)
    y_clean = true_forcing(x).detach()

def rms_forcing_residual_on_holdout(model, x_hold, y_hold):
    """
    RMS of u_xx(x_hold) - y_hold
    """
    model.eval()
    x = x_hold.detach().clone().requires_grad_(True)

    u = model(x)
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    r = u_xx - y_hold
    return torch.sqrt(torch.mean(r**2)).item()    


def compute_rms_forcing_residual(model, pde_problem, x_f):
    """
    RMS of forcing residual: u_xx(x_f) - y_f
    """
    model.eval()

    # Important: fresh tensor with gradients enabled
    x_f_ = x_f.detach().clone().requires_grad_(True)

    # Compute residual via PDE class
    residual = pde_problem.compute_residual(model, x_f_)

    rms = torch.sqrt(torch.mean(residual**2))
    return rms.item()
# =========================================================================
# Step 2: Main experiment
# =========================================================================
def run_hmc_unknown_noise():
    torch.manual_seed(42)

    # ------------------------------------------------------------------
    # Data setup — identical to train_bpinn.py
    # True solution: u(x) = sin(pi * x)
    # PDE: u_xx = -pi^2 * sin(pi * x)
    # ------------------------------------------------------------------

    # Boundary observations D_b: 2 sensors at x = -1 and x = 1
    # In paper notation: b̄^(i) = b(x_b^(i)) + epsilon_b, epsilon_b ~ N(0, sigma_b^2)
    x_b = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y_b = torch.tensor([[0.0],  [0.0]], dtype=torch.float32)
    y_b = y_b + torch.randn_like(y_b) * TRUE_SIGMA_B

    # Forcing term observations D_f: 20 sensors uniformly in [-1, 1]
    # In paper notation: f̄^(i) = f(x_f^(i)) + epsilon_f, epsilon_f ~ N(0, sigma_f^2)
    x_f = torch.linspace(-1, 1, 80).view(-1, 1).requires_grad_(True)
    y_f = -(torch.pi ** 2) * torch.sin(torch.pi * x_f).detach()
    y_f = y_f + torch.randn_like(y_f) * TRUE_SIGMA_F

    # PDE problem definition
    # Note: sigma_f argument in Poisson1D is not used by our extension
    # (sigma_f is inferred internally), but kept for API compatibility
    pde_problem = Poisson1D(x_f, y_f, sigma_f=TRUE_SIGMA_F)

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    model = BNN_UnknownNoise_f(input_dim=1, output_dim=1, hidden_dims=[20, 20])

    print("=" * 60)
    print("B-PINN with Unknown Forcing Term Noise (sigma_f)")
    print("=" * 60)
    print(f"Network parameters  : {model.num_params}")
    print(f"Total HMC dimension : {model.total_params}  "
          f"(network weights + log_sigma_f)")
    print(f"sigma_b             : {TRUE_SIGMA_B} (fixed and known)")
    print(f"sigma_f (true)      : {TRUE_SIGMA_F} (unknown — to be inferred)\n")

    # ------------------------------------------------------------------
    # Pretraining: bring weights to a good region before HMC
    # ------------------------------------------------------------------
    print("Step 1: Pretraining BNN weights via PINN loss (Adam, 2000 steps)...")
    pretrain_network(model, x_b, y_b, x_f, y_f, pde_problem, n_steps=2000)

    # ------------------------------------------------------------------
    # Build initial theta_full from pretrained weights + log_sigma_f guess
    # We initialize log_sigma_f = -2.3 => sigma_f = exp(-2.3) ≈ 0.1
    # This is our best guess based on domain knowledge.
    # The chain will explore around this starting point.
    # ------------------------------------------------------------------


    # à garder pour l'instant
    # rms_f = compute_rms_forcing_residual(model, pde_problem, x_f)
    # print("rms_f =", rms_f, "log =", np.log(rms_f))
    # sigma_f_init = max(0.1, 1.5 * rms_f)   # floor at true-ish scale + inflate a bit
    # theta_0 = model.get_initial_theta(log_sigma_f_init=np.log(sigma_f_init))

    theta_0 = model.get_initial_theta(log_sigma_f_init= -0.7)

    print(f"Initial log_sigma_f : {theta_0[model.num_params].item():.3f}  "
          f"=> sigma_f = {torch.exp(theta_0[model.num_params]).item():.4f}")
    print(f"True sigma_f        : {TRUE_SIGMA_F}\n")

    # ------------------------------------------------------------------
    # HMC sampling
    # sigma_b is passed as a fixed kwarg (as in original paper)
    # sigma_f is NOT passed — inferred from theta_full inside potential_energy
    # ------------------------------------------------------------------
    M       = 500    # posterior samples to keep
    N       = 2000   # total HMC iterations (burn-in = N - M = 1500)
    L       = 10     # leapfrog steps per iteration
    delta_t = 0.001  # leapfrog step size (small for stability post-pretraining)

    print(f"Step 2: HMC Sampling ({N} iterations, keeping last {M} samples)...")

    samples = HMC_sampler(
        model   = model,
        M       = M,
        N       = N,
        delta_t = delta_t,
        theta_0 = theta_0,
        L       = L,
        # Fixed kwargs passed to potential_energy:
        x_b         = x_b,
        y_b         = y_b,
        sigma_b     = TRUE_SIGMA_B,   # fixed and known
        x_f         = x_f,
        y_f         = y_f,
        pde_problem = pde_problem
        # Note: sigma_f is intentionally absent — inferred from theta_full
    )

    print(f"Sampling complete. Samples shape: {samples.shape}")

    # Chain mixing diagnostic
    theta_net_std = samples[:model.num_params, :].std(dim=1).mean().item()
    print(f"Chain mixing (mean std across weights): {theta_net_std:.5f}")
    print("  (should be > 0; if 0.00000 the chain is stuck)\n")

    # ------------------------------------------------------------------
    # Extract and report inferred sigma_f posterior
    # ------------------------------------------------------------------
    sigma_f_samples = model.extract_sigma_f_samples(samples)

    print("--- Inferred Noise Level ---")
    print(f"sigma_f | mean: {sigma_f_samples.mean().item():.4f}  "
          f"std: {sigma_f_samples.std().item():.4f}  "
          f"(true: {TRUE_SIGMA_F})")
    print(f"sigma_b | {TRUE_SIGMA_B} (fixed, not inferred)\n")

    # ------------------------------------------------------------------
    # Sigma trace plot (diagnostic: should look like noise, not flat line)
    # ------------------------------------------------------------------
    log_sigma_f_trace = samples[model.num_params, :].numpy()

    plt.figure(figsize=(8, 3))
    plt.plot(log_sigma_f_trace, color='orange', linewidth=1)
    plt.axhline(y=np.log(TRUE_SIGMA_F), color='black', linestyle='--',
                linewidth=2, label=f'True log(sigma_f) = {np.log(TRUE_SIGMA_F):.3f}')
    plt.title("log(sigma_f) trace — should look like noise, not a flat line")
    plt.xlabel("HMC sample index")
    plt.ylabel("log(sigma_f)")
    plt.legend()
    plt.tight_layout()
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/sigma_f_trace.png", dpi=150)
    print("Trace plot saved to experiments/results/sigma_f_trace.png")

    # ------------------------------------------------------------------
    # Main results plot
    # ------------------------------------------------------------------
    x_test = torch.linspace(-1, 1, 200).view(-1, 1)
    u_true = np.sin(np.pi * x_test.numpy())

    # Posterior predictive samples over u(x)
    u_preds = []
    for i in range(samples.shape[1]):
        theta_net = samples[:model.num_params, i]
        with torch.no_grad():
            u_pred = model.functional_forward(theta_net, x_test)
        u_preds.append(u_pred.numpy())

    u_preds = np.array(u_preds).squeeze()   # shape (M, 200)
    u_mean  = u_preds.mean(axis=0)
    u_std   = u_preds.std(axis=0)
    x_np    = x_test.numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: posterior predictive over u(x)
    ax = axes[0]
    ax.plot(x_np, u_true, 'k-',  label='True u(x)',       linewidth=2)
    ax.plot(x_np, u_mean, 'r--', label='Posterior mean',  linewidth=2)
    ax.fill_between(x_np,
                    u_mean - 2 * u_std,
                    u_mean + 2 * u_std,
                    alpha=0.3, color='cyan', label='±2 std')
    ax.scatter(x_b.numpy(), y_b.numpy(),
               c='blue', zorder=5, label='Boundary obs. (D_b)')
    ax.set_title("B-PINN Solution (sigma_f inferred, sigma_b fixed)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: posterior histogram over sigma_f
    ax = axes[1]
    ax.hist(sigma_f_samples.numpy(), bins=30,
            alpha=0.7, color='orange', density=True, label='Posterior sigma_f')
    ax.axvline(TRUE_SIGMA_F, color='black', linestyle='--',
               linewidth=2, label=f'True sigma_f = {TRUE_SIGMA_F}')
    ax.axvline(sigma_f_samples.mean().item(), color='red', linestyle='-',
               linewidth=2, label=f'Inferred mean = {sigma_f_samples.mean().item():.4f}')
    ax.set_title("Posterior over Inferred Forcing Term Noise sigma_f")
    ax.set_xlabel("sigma_f")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("experiments/results/bpinn_unknown_noise.png", dpi=150)
    print("Main plot saved to experiments/results/bpinn_unknown_noise.png")
    plt.show()


if __name__ == "__main__":
    run_hmc_unknown_noise()
