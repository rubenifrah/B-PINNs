import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import torch.nn.functional as F

class BNN(nn.Module):
    # =========================================================================
    # The BNN initialization tracks network architecture explicitly (input_dim, 
    # output_dim, hidden_dims) and calculates shape tracking (self.param_shapes).
    # Because for Bayesian Neural Networks using Hamiltonian Monte Carlo (HMC), 
    # we receive a 1D vector `theta` representing a proposed state of the model.
    # To evaluate the potential energy (and its gradient w.r.t theta), we cannot 
    # use PyTorch's in-place parameter updates (like `vector_to_parameters`) 
    # as it breaks the autograd computation graph. Instead, we must manually 
    # slice the 1D `theta` vector into weight and bias matrices during the forward pass.
    # Tracking these shapes during initialization is required for this runtime slicing.
    # =========================================================================
    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # Track shapes for slicing the 1D theta vector during functional forward
        self.param_shapes = []
        
        self.layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.param_shapes.append((h_dim, current_dim)) # Weight
            self.param_shapes.append((h_dim,))             # Bias
            current_dim = h_dim
            
        self.layers.append(nn.Linear(current_dim, output_dim))
        self.param_shapes.append((output_dim, current_dim)) # Weight
        self.param_shapes.append((output_dim,))             # Bias

        # Calculate total number of parameters to validate incoming theta vectors
        self.num_params = sum(torch.prod(torch.tensor(s)) for s in self.param_shapes)

    def forward(self, x):
        """Standard forward pass using internal parameters"""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def get_weights(self):
        """Returns the current weights of the network as a 1D vector"""
        return parameters_to_vector(self.parameters())
    
    def set_weights(self, theta):
        """Sets the weights of the network from a 1D vector"""
        vector_to_parameters(theta, self.parameters())

    # =========================================================================
    # Functional Forward Pass in addition to the standard forward pass
    # Standard PyTorch modules use their internally stored `.parameters()`.
    # When HMC proposes a new state `theta`, we need to compute the network's 
    # output using `theta` without detaching from PyTorch's autograd tracking. 
    # =========================================================================
    def functional_forward(self, theta, x):
        """
        Forward pass using explicit parameters theta without detaching the graph (no use of 
        vector_to_parameters)
        """
        if theta.numel() != self.num_params:
            raise ValueError(f"Expected theta size {self.num_params}, got {theta.numel()}")
            
        start = 0
        current_x = x
        num_layers = len(self.hidden_dims) + 1
        
        for i in range(num_layers):
            # Extract Weight matrix
            w_shape = self.param_shapes[2*i]
            w_numel = w_shape[0] * w_shape[1]
            weight = theta[start:start+w_numel].view(w_shape)
            start += w_numel
            
            # Extract Bias vector
            b_shape = self.param_shapes[2*i+1]
            b_numel = b_shape[0]
            bias = theta[start:start+b_numel]
            start += b_numel
            
            # Apply Linear transformation functionally
            current_x = F.linear(current_x, weight, bias)
            
            # Apply Activation (except for the last layer)
            if i < num_layers - 1:
                current_x = self.activation(current_x)
                
        return current_x

    def log_prior(self, theta, sigma_theta=1.0):
        # Log of the Gaussian prior: - (1/2 * sigma^2) * sum(theta^2)
        return -0.5 * torch.sum(theta**2) / (sigma_theta**2)
    
    # =========================================================================
    # We must compute `u_pred` and `u_f` using the functional forward pass so that 
    # the resulting likelihoods (and potential energy) maintain their gradient chain 
    # back to the `theta` vector proposed by HMC.
    # Here we also define a localized function `u_func_for_pde` to pass to the PDE problem.
    # =========================================================================

    def potential_energy(self, theta, x_u, y_u, x_f, y_f, sigma_u, sigma_f, pde_problem):
        """
        U(theta) = - [log p(data|theta) + log p(physics|theta) + log p(theta)]
        """
        # 1. Data Likelihood (MSE on measurements)
        # MUST use functional_forward to keep theta in the computation graph
        u_pred = self.functional_forward(theta, x_u)
        log_lik_u = -0.5 * torch.sum((u_pred - y_u)**2) / (sigma_u**2)
        
        # 2. Physics Likelihood (The PDE residual)
        x_f.requires_grad_(True)
        # Create a proxy function for the PDE module to compute spatial derivatives
        # This function wraps the functional forward so the PDE evaluator doesn't
        # need to know about theta handling.
        def u_func_for_pde(x):
            return self.functional_forward(theta, x)
        
        # Compute the residual using the PDE problem definition
        res_f = pde_problem.compute_residual(u_func_for_pde, x_f)
        log_lik_f = -0.5 * torch.sum(res_f**2) / (sigma_f**2)

        # 3. Parameter Prior
        log_p = self.log_prior(theta)
        
        # Total U(theta) is the negative log posterior
        return -(log_lik_u + log_lik_f + log_p)
    
    def hamiltonian(self, theta, r, **kwargs):
        """
        Computes the Hamiltonian H(theta, r) = U(theta) + K(r)
        Assuming Identity Mass matrix, K(r) = 0.5 * r^T r
        """
        U = self.potential_energy(theta, **kwargs)
        K = 0.5 * torch.sum(r**2)
        return U + K

    def gradient(self, theta, **kwargs):
        """
        Compute gradient of Potential Energy U w.r.t theta.
        Matches the signautre expected by HMC sampler.
        """
        theta_copy = theta.clone().detach().requires_grad_(True)
        U = self.potential_energy(theta_copy, **kwargs)
        grad = torch.autograd.grad(U, theta_copy)[0]
        return grad