import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

class BNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh()):
        super().__init__()
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.activation = activation

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def get_weights(self):
        """Returns all parameters as a single 1D vector theta."""
        return parameters_to_vector(self.parameters())

    def set_weights(self, theta):
        """Sets the network parameters from a 1D vector theta."""
        vector_to_parameters(theta, self.parameters())

    def log_prior(self, theta, sigma_theta=1.0):
        # Log of the Gaussian prior: - (1/2 * sigma^2) * sum(theta^2)
        return -0.5 * torch.sum(theta**2) / (sigma_theta**2)
    
    def potential_energy(self, theta, x_u, y_u, x_f, y_f, sigma_u, sigma_f, pde_problem):
        """
        U(theta) = - [log p(data|theta) + log p(physics|theta) + log p(theta)]
        """
        self.set_weights(theta)
        
        # 1. Data Likelihood (MSE on measurements)
        u_pred = self.forward(x_u)
        log_lik_u = -0.5 * torch.sum((u_pred - y_u)**2) / (sigma_u**2)
        
        # 2. Physics Likelihood (The PDE residual)
        # This requires autograd to compute derivatives of u w.r.t x_f
        x_f.requires_grad_(True)
        u_f = self.forward(x_f)
        
        # Compute the residual using the PDE problem
        res_f = pde_problem.compute_residual(self.forward, x_f)
        log_lik_f = -0.5 * torch.sum(res_f**2) / (sigma_f**2)

        log_p = self.log_prior(theta)
        
        # 3. Total U(theta)
        return -(log_lik_u + log_lik_f + log_p)
    
    def get_gradient(self, theta, **kwargs):
        theta_copy = theta.clone().detach().requires_grad_(True)
        U = self.potential_energy(theta_copy, **kwargs)
        grad = torch.autograd.grad(U, theta_copy)[0]
        return grad