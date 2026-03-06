import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

class BNN_UnknownNoise(nn.Module):
    """
    Extension of the original BNN for B-PINNs where noise levels sigma_u and sigma_f
    are treated as unknown parameters and inferred jointly with the network weights.

    The key idea is to augment the HMC parameter vector theta with two extra scalar
    parameters: log_sigma_u and log_sigma_f. Working in log-space ensures sigma > 0
    throughout sampling without any constrained optimization.

    The full sampled vector is therefore:
        theta_full = [theta_network (num_params,), log_sigma_u (1,), log_sigma_f (1,)]

    This requires NO changes to the HMC sampler, since it is agnostic to what theta contains.
    Only potential_energy() and gradient() need to be aware of the new structure.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh(), mu_log_sigma=-2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.mu_log_sigma = mu_log_sigma

        # Track shapes for slicing the 1D theta vector during functional forward
        # (identical to original BNN)
        self.param_shapes = []

        self.layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.param_shapes.append((h_dim, current_dim))  # Weight
            self.param_shapes.append((h_dim,))              # Bias
            current_dim = h_dim

        self.layers.append(nn.Linear(current_dim, output_dim))
        self.param_shapes.append((output_dim, current_dim))  # Weight
        self.param_shapes.append((output_dim,))               # Bias

        # Number of network parameters only (NOT including log_sigmas)
        self.num_params = sum(torch.prod(torch.tensor(s)) for s in self.param_shapes)

        # Total dimension of the HMC vector = network params + log_sigma_u + log_sigma_f
        self.total_params = self.num_params + 1

    # =========================================================================
    # Helper: split the full HMC vector into network weights and log-sigmas
    # =========================================================================
    def split_theta(self, theta_full):
        """Splits the vector into network weights and just log_sigma_f"""
        theta_net = theta_full[:self.num_params]
        log_sigma_f = theta_full[self.num_params] # Just one extra param
        return theta_net, log_sigma_f

    def get_initial_theta(self, log_sigma_f_init=-2.3):
        """Only initialize log_sigma_f"""
        theta_net = parameters_to_vector(self.parameters()).detach()
        log_sigma_f = torch.tensor([log_sigma_f_init], dtype=torch.float32)
        return torch.cat([theta_net, log_sigma_f])

    # =========================================================================
    # Functional forward pass — identical to original BNN, operates on theta_net only
    # =========================================================================
    def forward(self, x):
        """Standard forward pass using internal parameters."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def functional_forward(self, theta_net, x):
        """
        Forward pass using explicit network weights theta_net.
        Keeps theta_net in the autograd computation graph (no vector_to_parameters).
        """
        if theta_net.numel() != self.num_params:
            raise ValueError(f"Expected theta_net size {self.num_params}, got {theta_net.numel()}")

        start = 0
        current_x = x
        num_layers = len(self.hidden_dims) + 1

        for i in range(num_layers):
            w_shape = self.param_shapes[2 * i]
            w_numel = w_shape[0] * w_shape[1]
            weight = theta_net[start:start + w_numel].view(w_shape)
            start += w_numel

            b_shape = self.param_shapes[2 * i + 1]
            b_numel = b_shape[0]
            bias = theta_net[start:start + b_numel]
            start += b_numel

            current_x = F.linear(current_x, weight, bias)

            if i < num_layers - 1:
                current_x = self.activation(current_x)

        return current_x

    # =========================================================================
    # Priors
    # =========================================================================
    def log_prior_theta(self, theta_net, sigma_theta=1.0):
        """Standard Gaussian prior on network weights."""
        return -0.5 * torch.sum(theta_net ** 2) / (sigma_theta ** 2)

    def log_prior_sigma(self, log_sigma):
        mu_log_sigma = self.mu_log_sigma
        tau = 0.5            # flexibility 
        return -0.5 * ((log_sigma - mu_log_sigma) / tau) ** 2

    # =========================================================================
    # Potential energy — the core of the extension
    # =========================================================================
    def potential_energy(self, theta_full, x_u, y_u, x_f, y_f, pde_problem, sigma_u):
        
        # 1. Split the augmented vector (only extracting log_sigma_f)
        theta_net, log_sigma_f = self.split_theta(theta_full)

        # 2. Recover sigma_f
        sigma_f = torch.exp(log_sigma_f)

        # 3. Data likelihood (Uses the FIXED sigma_u passed as an argument)
        u_pred = self.functional_forward(theta_net, x_u)
        N_u = y_u.shape[0]
        # Notice we don't have -N_u * log_sigma_u here because it's a constant
        log_lik_u = -0.5 * torch.sum((u_pred - y_u) ** 2) / (sigma_u ** 2)

        # 4. Physics likelihood (Uses the INFERRED sigma_f)
        x_f.requires_grad_(True)
        def u_func_for_pde(x):
            return self.functional_forward(theta_net, x)

        res_f = pde_problem.compute_residual(u_func_for_pde, x_f)
        N_f = res_f.shape[0]
        log_lik_f = -N_f * log_sigma_f - 0.5 * torch.sum(res_f ** 2) / (sigma_f ** 2)

        # 5. Priors
        log_p_theta = self.log_prior_theta(theta_net)
        log_p_sigma_f = self.log_prior_sigma(log_sigma_f) # Prior only on sigma_f

        # 6. Total potential energy
        log_posterior = log_lik_u + log_lik_f + log_p_theta + log_p_sigma_f
        return -log_posterior

    def hamiltonian(self, theta_full, r, **kwargs):
        """
        H(theta_full, r) = U(theta_full) + 0.5 * r^T r
        Identical structure to original BNN — HMC sampler calls this unchanged.
        """
        U = self.potential_energy(theta_full, **kwargs)
        K = 0.5 * torch.sum(r ** 2)
        return U + K

    def gradient(self, theta_full, **kwargs):
        """
        Gradient of U w.r.t theta_full (including log_sigma dimensions).
        Called by HMC sampler — signature unchanged.
        """
        theta_copy = theta_full.clone().detach().requires_grad_(True)
        U = self.potential_energy(theta_copy, **kwargs)
        grad = torch.autograd.grad(U, theta_copy)[0]
        return grad

    # =========================================================================
    # Utility: extract inferred sigma statistics from posterior samples
    # =========================================================================
    def extract_sigma_samples(self, samples):
        """Extract only sigma_f"""
        log_sigma_f_samples = samples[self.num_params, :]
        return torch.exp(log_sigma_f_samples)
