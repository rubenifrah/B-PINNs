import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim
        self.layers.append(nn.Linear(current_dim, output_dim))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def get_weights(self):
        return parameters_to_vector(self.parameters())
    
    def set_weights(self, theta):
        vector_to_parameters(theta, self.parameters())
    
# =========================================================================
# PINN baseline to compare against the B-PINN.
# This provides the exact structure needed by `pde_problem.compute_residual` 
# that the B-PINN uses, ensuring a fair baseline comparison using standard optimization (e.g. Adam/L-BFGS).
# =========================================================================
class PINN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh()):
        super().__init__()
        self.net = MLP(input_dim, hidden_dims, output_dim, activation)

    def forward(self, x):
        """Standard forward pass proxying the internal MLP."""
        return self.net(x)

    def compute_loss(self, x_u, y_u, x_f, pde_problem):
        """
        Computes the total loss for standard PINN training: Loss_data + Loss_physics
        """
        # 1. Data Loss
        # If there's no data (pure forward problem depending on physics), x_u/y_u might be empty.
        if x_u is not None and len(x_u) > 0:
            u_pred = self.forward(x_u)
            mse_u = torch.mean((u_pred - y_u)**2)
        else:
            mse_u = torch.tensor(0.0, device=x_f.device)

        # 2. Physics Loss
        x_f.requires_grad_(True)
        # We pass self.forward to the PDE problem to compute derivatives w.r.t x_f
        res_f = pde_problem.compute_residual(self.forward, x_f)
        mse_f = torch.mean(res_f**2)
        
        # Total Loss is the sum of component MSEs
        total_loss = mse_u + mse_f
        
        return total_loss, mse_u, mse_f