import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.BNN_unknown_noise import BNN_UnknownNoise
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs import Poisson1D

from src.models.PINN import PINN
from src.physics.PDEs import Poisson1D
import torch.optim as optim

def pretrain_network(model, x_u, y_u, x_f, y_f, pde_problem, n_steps=2000):
    """
    Pretrain the BNN weights using standard PINN loss before HMC.
    This gives HMC a much better starting point in weight space.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        # Data loss
        u_pred = model.forward(x_u)
        loss_u = torch.mean((u_pred - y_u)**2)
        
        # Physics loss
        x_f_grad = x_f.clone().requires_grad_(True)
        res = pde_problem.compute_residual(model.forward, x_f_grad)
        loss_f = torch.mean(res**2)
        
        loss = loss_u + loss_f
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Pretrain step {step}, loss: {loss.item():.6f}")
    
    print(f"  Pretraining done. Final loss: {loss.item():.6f}")

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

TRUE_SIGMA_U = 0.1
TRUE_SIGMA_F = 0.1

def run_hmc_unknown_noise():

    # ------------------------------------------------------------------
    # 1. Generate data — identical setup to train_bpinn.py
    # ------------------------------------------------------------------
    torch.manual_seed(42)

    # Boundary / solution data
    x_u = torch.tensor([[-1.0], [1.0]], dtype=torch.float32)
    y_u = torch.tensor([[0.0], [0.0]], dtype=torch.float32)
    y_u = y_u + torch.randn_like(y_u) * TRUE_SIGMA_U

    # Collocation / forcing term data
    x_f = torch.linspace(-1, 1, 20).view(-1, 1).requires_grad_(True)
    y_f = -(torch.pi ** 2) * torch.sin(torch.pi * x_f).detach()
    y_f = y_f + torch.randn_like(y_f) * TRUE_SIGMA_F

    # PDE problem — sigma_f argument is kept for compatibility but will NOT
    # be used as a fixed value (the model infers it instead)
    pde_problem = Poisson1D(x_f, y_f, sigma_f=None)

    # ------------------------------------------------------------------
    # 2. Setup model
    # ------------------------------------------------------------------
    model = BNN_UnknownNoise(input_dim=1, output_dim=1, hidden_dims=[20, 20])

    print(f"Network parameters:  {model.num_params}")
    print(f"Total HMC dimension: {model.total_params}  (+ log_sigma_u, log_sigma_f)")

    # Pretrain first
    print("Pretraining BNN weights via PINN loss...")
    pretrain_network(model, x_u, y_u, x_f, y_f, pde_problem)

    # Then initialize theta_0 from pretrained weights
    theta_0 = model.get_initial_theta(
        log_sigma_u_init=-2.3,  # exp(-2.3) ≈ 0.1
        log_sigma_f_init=-2.3
    )

    # ------------------------------------------------------------------
    # 3. Initialize theta_full
    # We start log_sigma at 0.0 => sigma = 1.0 (intentionally wrong)
    # to show the sampler recovers the true value.
    # Alternatively use log(0.5) as a closer starting point.
    # ------------------------------------------------------------------
    # theta_0 = model.get_initial_theta(
    # log_sigma_u_init=-2.0,
    # log_sigma_f_init=-2.0
    # )
    # theta_0 = model.get_initial_theta(
    #     log_sigma_u_init=0.0,   # starting guess: sigma_u = 1.0 (true: 0.1)
    #     log_sigma_f_init=0.0    # starting guess: sigma_f = 1.0 (true: 0.1)
    # )

    print(f"\nInitial log_sigma_u: {theta_0[model.num_params].item():.3f}  "
          f"=> sigma_u = {torch.exp(theta_0[model.num_params]).item():.3f}")
    print(f"Initial log_sigma_f: {theta_0[model.num_params+1].item():.3f}  "
          f"=> sigma_f = {torch.exp(theta_0[model.num_params+1]).item():.3f}")
    print(f"True sigma_u = {TRUE_SIGMA_U}, True sigma_f = {TRUE_SIGMA_F}\n")

    # ------------------------------------------------------------------
    # 4. HMC parameters
    # NOTE: sigma_u and sigma_f are NO LONGER passed as kwargs —
    # they are inferred from theta_full inside potential_energy().
    # ------------------------------------------------------------------
    N       = 2000   # was 300
    M       = 500    # was 100
    L       = 10     # was 20
    delta_t = 0.001  # slightly smaller
    # M       = 100    # samples to keep
    # N       = 300    # total HMC iterations (more than baseline due to extra dims)
    # L       = 20     # leapfrog steps
    # delta_t = 0.005  # slightly smaller step size due to higher dimensionality

    print(f"Starting HMC with {N} iterations, keeping last {M} samples...")

    samples = HMC_sampler(
        model=model,
        M=M,
        N=N,
        delta_t=delta_t,
        theta_0=theta_0,
        L=L,
        # kwargs passed to potential_energy — note: no sigma_u / sigma_f here
        x_u=x_u,
        y_u=y_u,
        x_f=x_f,
        y_f=y_f,
        pde_problem=pde_problem
    )

    # Check acceptance rate proxy — if std is near zero, chain is stuck
    theta_net_samples = samples[:model.num_params, :]
    print(f"Chain mixing check - mean std across weights: "
        f"{theta_net_samples.std(dim=1).mean().item():.5f}")

    # Also plot the sigma trace to see if it moved
    plt.figure()
    plt.plot(samples[model.num_params, :].numpy(), label='log_sigma_u trace')
    plt.plot(samples[model.num_params+1, :].numpy(), label='log_sigma_f trace')
    plt.legend()
    plt.title("Sigma trace — should look like noise, not a flat line")
    plt.savefig("experiments/results/sigma_trace.png")

    print(f"Sampling complete. Samples shape: {samples.shape}")

    # ------------------------------------------------------------------
    # 5. Extract and report inferred noise levels
    # ------------------------------------------------------------------
    sigma_u_samples, sigma_f_samples = model.extract_sigma_samples(samples)

    print("\n--- Inferred Noise Levels ---")
    print(f"sigma_u | mean: {sigma_u_samples.mean().item():.4f}  "
          f"std: {sigma_u_samples.std().item():.4f}  "
          f"(true: {TRUE_SIGMA_U})")
    print(f"sigma_f | mean: {sigma_f_samples.mean().item():.4f}  "
          f"std: {sigma_f_samples.std().item():.4f}  "
          f"(true: {TRUE_SIGMA_F})")

    # ------------------------------------------------------------------
    # 6. Plot results
    # ------------------------------------------------------------------
    x_test = torch.linspace(-1, 1, 200).view(-1, 1)
    u_true = np.sin(np.pi * x_test.numpy())

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
    ax.scatter(x_u.numpy(), y_u.numpy(), c='blue', zorder=5, label='Noisy observations')
    ax.set_title("B-PINN Solution (Unknown Noise)")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Posterior distributions over sigma_u and sigma_f
    ax = axes[1]
    ax.hist(sigma_u_samples.numpy(), bins=20, alpha=0.6,
            color='blue', label=f'σ_u  (true={TRUE_SIGMA_U})', density=True)
    ax.hist(sigma_f_samples.numpy(), bins=20, alpha=0.6,
            color='orange', label=f'σ_f  (true={TRUE_SIGMA_F})', density=True)
    ax.axvline(TRUE_SIGMA_U, color='blue', linestyle='--', linewidth=2)
    ax.axvline(TRUE_SIGMA_F, color='orange', linestyle='--', linewidth=2)
    ax.set_title("Posterior over Inferred Noise Levels")
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
