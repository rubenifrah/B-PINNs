"""
Diagnostic script: Track BNN parameter norm during HMC.
If HMC is fitting the wave by abandoning the prior, the L2 norm
of the weights will blow up far beyond the expected prior norm.

For a BNN with N_params, if theta ~ N(0, 1), then
E[||theta||^2] = N_params. So E[||theta||] approx sqrt(N_params).
Let's see what the HMC sampler actually produces to fit the data.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.BNN import BNN
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs import PDEProblem

torch.manual_seed(42)

# Same setup as the test script
lam = 0.01
x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
y_b = torch.sin(6 * x_b)**3

x_f = torch.linspace(-0.7, 0.7, 16).view(-1, 1).requires_grad_(True)
y_f = 0.01 * 108.0 * (2 * torch.sin(6 * x_f) * (torch.cos(6 * x_f) ** 2) - (torch.sin(6 * x_f) ** 3)).detach()

class ScaledPoisson1D(PDEProblem):
    def compute_residual(self, u_func, x, params=None):
        u = u_func(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return 0.01 * u_xx - y_f
pde_problem = ScaledPoisson1D(x_f, y_f, 0.01)

# Initialize standard BNN
bnn = BNN(input_dim=1, output_dim=1, hidden_dims=[50, 50], prior_std=1.0)
expected_norm = np.sqrt(bnn.num_params)
print(f"Number of parameters: {bnn.num_params}")
print(f"Expected L2 norm under prior N(0, 1): ~{expected_norm:.2f}")

# HMC Sampling (Case 1: sigma=0.01)
samples = HMC_sampler(
    model=bnn,
    M=100, N=1000, delta_t=0.0001, theta_0=bnn.get_weights(), L=50,
    x_b=x_b, y_b=y_b, x_f=x_f, y_f=y_f, sigma_u=0.01, sigma_f=0.01,
    pde_problem=pde_problem, x_u=None, y_u=None
)

# Calculate L2 norms of the sampled networks
sample_norms = torch.norm(samples, dim=0).detach().numpy()

plt.figure(figsize=(10, 6))
plt.plot(sample_norms, 'b-', label='HMC Sample L2 Norms')
plt.axhline(y=expected_norm, color='r', linestyle='--', label='Expected Prior L2 Norm')
plt.xlabel('HMC Steps (Last 100)')
plt.ylabel('L2 Norm of $\\theta$')
plt.title('Weight Blowup: HMC abandons the prior to fit the data')
plt.legend()
plt.savefig('diagnostic_norm.png')
print("Saved diagnostic_norm.png")
