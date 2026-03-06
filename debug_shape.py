import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models.BNN import BNN
from src.utils.training import train_bpinn
from src.physics.PDEs import ScaledPoisson1D

w = 10
dim = [1, w, w, 1]
bnn = BNN(input_dim=1, output_dim=1, hidden_dims=[w, w], prior_std=1.0)

x_b = torch.tensor([[-0.7], [0.7]], dtype=torch.float32)
y_b = torch.sin(6 * x_b) ** 3
x_f = torch.linspace(-0.7, 0.7, 16).view(-1, 1)
y_f = 0.01 * 108.0 * (2 * torch.sin(6 * x_f) * (torch.cos(6 * x_f) ** 2) - (torch.sin(6 * x_f) ** 3))

pde_problem = ScaledPoisson1D(x_f, y_f, 0.1, lambda_param=0.01)

samples = train_bpinn(
    model=bnn,
    pde_problem=pde_problem,
    x_b=x_b, y_b=y_b,
    x_f=x_f, y_f=y_f,
    N=5,
    L=5,
    delta_t=0.005,
    theta_0=None,
    burn_in=2
)
print(f"Type of samples: {type(samples)}")
if isinstance(samples, list):
    print(f"Length of list: {len(samples)}")
    print(f"Type of element 0: {type(samples[0])}")
    if hasattr(samples[0], 'shape'):
        print(f"Shape of element 0: {samples[0].shape}")
elif hasattr(samples, 'shape'):
    print(f"Shape of samples: {samples.shape}")
