import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.models.PIGP import ExactPIGP
from src.models.NNGP import Erf_NNGP_kernel
from src.models.BNN import BNN
from src.utils.training import train_bpinn
from src.physics.PDEs import ScaledPoisson1D

# Set up test metrics
sigma_w2 = 1.0
sigma_b2 = 1.0
depth = 2 # Maps to [width, width] hidden layers in finite case
noise_scale = 0.1
torch.manual_seed(42)

# Generate identical 1D Poisson problem data
x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
y_b = torch.sin(6 * x_b) ** 3
y_b += torch.randn_like(y_b) * noise_scale

x_f = torch.linspace(-0.7, 0.7, 16).view(-1, 1)
y_f = 0.01 * 108.0 * (2 * torch.sin(6 * x_f) * (torch.cos(6 * x_f) ** 2) - (torch.sin(6 * x_f) ** 3))
y_f += torch.randn_like(y_f) * noise_scale

# 1. Exact PI-GP Posterior (Infinite Width NNGP Limit)
print("Evaluating Infinite-Width NNGP Limit...")
def my_kernel(x, xp):
    x, xp = x.squeeze(), xp.squeeze()
    return Erf_NNGP_kernel(x, xp, depth=depth, sigma_w2=sigma_w2, sigma_b2=sigma_b2)

pigp = ExactPIGP(kernel_fn=my_kernel, lambda_pde=0.01, noise_u=noise_scale, noise_f=noise_scale)
X_test = torch.linspace(-0.8, 0.8, 100).view(-1, 1)

mu_gp, var_gp = pigp.fit_and_predict(x_b, y_b, x_f, y_f, X_test)
std_gp = torch.sqrt(var_gp).detach().numpy().flatten()
mu_gp = mu_gp.detach().numpy().flatten()

# 2. HMC Sampling over Finite Widths
widths = [10, 50, 100]
results = []
# Match GP prior with BNN prior:
# We scale the prior variance internally by 1/sqrt(width). BNN base standard normal is scaled.
prior_std = np.sqrt(sigma_w2)

pde_problem = ScaledPoisson1D(x_f, y_f, noise_scale, lambda_param=0.01)

for w in widths:
    print(f"\nEvaluating Finite Width BNN: {w}x{w}...")
    bnn = BNN(input_dim=1, output_dim=1, hidden_dims=[w, w], prior_std=prior_std)
    
    # Run HMC Sampler
    samples = train_bpinn(
        model=bnn,
        pde_problem=pde_problem,
        x_b=x_b, y_b=y_b,
        x_f=x_f, y_f=y_f,
        N=500,  # Lowered slightly for speed loop
        L=50,
        delta_t=0.005,
        theta_0=None,
        burn_in=100
    )
    
    # Evaluate predictions
    preds = []
    # HMC returns shape [num_params, num_samples]. Transpose to [num_samples, num_params]
    samples = samples.T
    
    for i in range(samples.shape[0]):
        theta = samples[i]
        preds.append(bnn.functional_forward(theta, X_test).detach())
        
    preds = torch.stack(preds)
    mu_bnn = preds.mean(dim=0).numpy().flatten()
    std_bnn = preds.std(dim=0).numpy().flatten()
    results.append((w, mu_bnn, std_bnn))

# 3. Plotting Grid
print("\nPlotting Convergence...")
x_t = X_test.numpy().flatten()
u_true = (np.sin(6 * x_t) ** 3)

fig, axes = plt.subplots(1, len(widths) + 1, figsize=(18, 5), sharey=True)

# Plot GP
axes[0].plot(x_t, u_true, 'k--', label='True u(x)')
axes[0].plot(x_t, mu_gp, 'b-', label='Exact GP Mean')
axes[0].fill_between(x_t, mu_gp - 2*std_gp, mu_gp + 2*std_gp, color='blue', alpha=0.2)
axes[0].scatter(x_b.numpy(), y_b.numpy(), color='red', zorder=5)
axes[0].set_title(r'Exact PI-GP $(N \to \infty)$')
axes[0].legend()

# Plot finite widths
for i, (w, mu_bnn, std_bnn) in enumerate(results):
    axes[i+1].plot(x_t, u_true, 'k--')
    axes[i+1].plot(x_t, mu_bnn, 'g-')
    axes[i+1].fill_between(x_t, mu_bnn - 2*std_bnn, mu_bnn + 2*std_bnn, color='green', alpha=0.2)
    axes[i+1].scatter(x_b.numpy(), y_b.numpy(), color='red', zorder=5)
    axes[i+1].set_title(f'HMC B-PINN (Width: {w})')

plt.tight_layout()
plt.savefig('convergence_benchmark.png')
print("Successfully saved convergence_benchmark.png")
