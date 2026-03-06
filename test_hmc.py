import torch
import sys
import os
import matplotlib.pyplot as plt
from src.models.PINN import PINN
from src.models.BNN import BNN
from src.physics.PDEs import PDEProblem
from src.utils.training import train_pinn, train_bpinn

class ScaledPoisson1D(PDEProblem):
    def compute_residual(self, u_func, x, params=None):
        u = u_func(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return 0.01 * u_xx - self.y_f

x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
y_b = torch.sin(6 * x_b) ** 3
x_f = torch.linspace(-0.7, 0.7, 16).view(-1, 1).requires_grad_(True)
y_f = 0.01 * 108.0 * (2 * torch.sin(6 * x_f) * (torch.cos(6 * x_f) ** 2) - (torch.sin(6 * x_f) ** 3)).detach()

noise_scale = 0.1
torch.manual_seed(42)
y_b = y_b + torch.randn_like(y_b) * noise_scale
y_f = y_f + torch.randn_like(y_f) * noise_scale

pde_problem = ScaledPoisson1D(x_f, y_f, noise_scale)
bnn_model = BNN(input_dim=1, output_dim=1, hidden_dims=[50, 50])

print("Testing delta_t=0.005...")
try:
    samples = train_bpinn(
        model=bnn_model, pde_problem=pde_problem,
        x_b=x_b, y_b=y_b, x_f=x_f, y_f=y_f,
        sigma_u=noise_scale, sigma_f=noise_scale,
        M=10, N=20, L=50, delta_t=0.005, theta_0=None
    )
except Exception as e:
    print("Error:", e)
