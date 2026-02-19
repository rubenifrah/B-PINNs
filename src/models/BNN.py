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