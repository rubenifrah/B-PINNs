import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F


class BNN_Inverse(nn.Module):
    """
    B-PINN for the inverse problem of Section 3.3.1 (Yang et al., 2020).

    PDE: lambda * u_xx + k * tanh(u) = f,   x in [-0.7, 0.7]

    The unknown PDE parameter k is inferred jointly with the BNN weights theta
    via HMC, following Equation 7 of the paper:

        P(theta, k | D) proportional to P(D | theta, k) * P(theta) * P(k)

    Augmented HMC vector:
        theta_full = [theta_net (num_params,), k (1,)]

    Dataset (Section 3.3.1):
        D_u : 6 interior sensors  — direct observations of u(x)
        D_b : 2 boundary sensors  — direct observations of u at x = +-0.7
        D_f : 32 sensors          — noisy measurements of forcing term f(x)

    All noise levels sigma_u, sigma_f, sigma_b are KNOWN and FIXED.
    Only k is unknown.
    """

    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh()):
        super().__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.hidden_dims = hidden_dims
        self.activation  = activation

        self.param_shapes = []
        self.layers       = nn.ModuleList()
        current_dim       = input_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.param_shapes.append((h_dim, current_dim))
            self.param_shapes.append((h_dim,))
            current_dim = h_dim

        self.layers.append(nn.Linear(current_dim, output_dim))
        self.param_shapes.append((output_dim, current_dim))
        self.param_shapes.append((output_dim,))

        self.num_params   = sum(torch.prod(torch.tensor(s)) for s in self.param_shapes)
        self.total_params = self.num_params + 1  # + k

    # =========================================================================
    # Split / build theta_full
    # =========================================================================
    def split_theta(self, theta_full):
        if theta_full.numel() != self.total_params:
            raise ValueError(
                f"Expected theta_full size {self.total_params}, "
                f"got {theta_full.numel()}."
            )
        theta_net = theta_full[:self.num_params]
        k         = theta_full[self.num_params:self.num_params + 1]
        return theta_net, k

    def get_initial_theta(self, k_init=0.5):
        """
        Builds the initial HMC vector from pretrained network weights + k_init.
        k_init is set deliberately away from true value (0.7) to test inference.
        """
        theta_net = parameters_to_vector(self.parameters()).detach()
        k         = torch.tensor([k_init], dtype=torch.float32)
        return torch.cat([theta_net, k])

    # =========================================================================
    # Forward passes
    # =========================================================================
    def forward(self, x):
        """Standard forward pass. Used during pretraining."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def functional_forward(self, theta_net, x):
        """
        Autograd-compatible forward pass using explicit theta_net.
        Keeps gradients flowing to theta_full inside potential_energy.
        """
        if theta_net.numel() != self.num_params:
            raise ValueError(
                f"Expected theta_net size {self.num_params}, got {theta_net.numel()}"
            )
        start      = 0
        current_x  = x
        num_layers = len(self.hidden_dims) + 1

        for i in range(num_layers):
            w_shape   = self.param_shapes[2 * i]
            w_numel   = w_shape[0] * w_shape[1]
            weight    = theta_net[start:start + w_numel].view(w_shape)
            start    += w_numel
            b_shape   = self.param_shapes[2 * i + 1]
            b_numel   = b_shape[0]
            bias      = theta_net[start:start + b_numel]
            start    += b_numel
            current_x = F.linear(current_x, weight, bias)
            if i < num_layers - 1:
                current_x = self.activation(current_x)

        return current_x

    # =========================================================================
    # Priors
    # =========================================================================
    def log_prior_theta(self, theta_net, sigma_theta=1.0):
        """Gaussian prior on network weights."""
        return -0.5 * torch.sum(theta_net ** 2) / (sigma_theta ** 2)

    def log_prior_k(self, k, sigma_k=1.0):
        """Gaussian prior on k — weakly informative."""
        return -0.5 * torch.sum(k ** 2) / (sigma_k ** 2)

    # =========================================================================
    # Potential energy
    # =========================================================================
    def potential_energy(self, theta_full, x_u, y_u, x_b, y_b,
                         x_f, y_f, sigma_u, sigma_b, sigma_f, pde_problem):
        """
        U(theta_full) = - log P(D_u | theta, k)
                        - log P(D_b | theta, k)
                        - log P(D_f | theta, k)
                        - log P(theta)
                        - log P(k)
        """
        theta_net, k = self.split_theta(theta_full)

        # 1. Interior u likelihood — direct observation of u
        u_pred    = self.functional_forward(theta_net, x_u)
        log_lik_u = -0.5 * torch.sum((u_pred - y_u) ** 2) / (sigma_u ** 2)

        # 2. Boundary likelihood — direct observation of u at boundary
        b_pred    = self.functional_forward(theta_net, x_b)
        log_lik_b = -0.5 * torch.sum((b_pred - y_b) ** 2) / (sigma_b ** 2)

        # 3. Forcing term likelihood — PDE residual with k passed explicitly
        x_f.requires_grad_(True)

        def u_func_for_pde(x):
            return self.functional_forward(theta_net, x)

        res_f     = pde_problem.compute_residual(u_func_for_pde, x_f, params=k)
        log_lik_f = -0.5 * torch.sum(res_f ** 2) / (sigma_f ** 2)

        # 4. Priors
        log_p_theta = self.log_prior_theta(theta_net)
        log_p_k     = self.log_prior_k(k)

        return -(log_lik_u + log_lik_b + log_lik_f + log_p_theta + log_p_k)

    # =========================================================================
    # Hamiltonian and gradient — called by HMC sampler unchanged
    # =========================================================================
    def hamiltonian(self, theta_full, r, **kwargs):
        U = self.potential_energy(theta_full, **kwargs)
        K = 0.5 * torch.sum(r ** 2)
        return U + K

    def gradient(self, theta_full, **kwargs):
        theta_copy = theta_full.clone().detach().requires_grad_(True)
        U          = self.potential_energy(theta_copy, **kwargs)
        grad       = torch.autograd.grad(U, theta_copy)[0]
        return grad

    # =========================================================================
    # Utilities
    # =========================================================================
    def extract_k_samples(self, samples):
        """Returns posterior samples of k, shape (M,)"""
        return samples[self.num_params, :]

    def predict_f(self, theta_net, x, k_val, lambda_val):
        """
        Computes f̃(x) = lambda*u_xx + k*tanh(u) for a single posterior sample.
        Used for plotting the posterior predictive over f(x).

        Args:
            theta_net : network weights for this sample
            x         : test points, shape (N, 1)
            k_val     : scalar k value for this sample
            lambda_val: known PDE coefficient lambda
        """
        x_in = x.clone().detach().requires_grad_(True)
        u    = self.functional_forward(theta_net, x_in)
        u_x  = torch.autograd.grad(
            u, x_in, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x_in, grad_outputs=torch.ones_like(u_x), create_graph=False)[0]
        f_pred = lambda_val * u_xx + k_val * torch.tanh(u.detach())
        return f_pred.detach()