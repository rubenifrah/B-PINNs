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

    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

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
        self.total_params = self.num_params + 2

    # =========================================================================
    # Helper: split the full HMC vector into network weights and log-sigmas
    # =========================================================================
    def split_theta(self, theta_full):
        """
        Splits the augmented HMC vector into:
            - theta_net: network weights (num_params,)
            - log_sigma_u: scalar (1,)
            - log_sigma_f: scalar (1,)
        """
        if theta_full.numel() != self.total_params:
            raise ValueError(
                f"Expected theta_full size {self.total_params}, got {theta_full.numel()}. "
                f"Remember: theta_full = [network weights ({self.num_params}), "
                f"log_sigma_u (1), log_sigma_f (1)]"
            )
        theta_net = theta_full[:self.num_params]
        log_sigma_u = theta_full[self.num_params]
        log_sigma_f = theta_full[self.num_params + 1]
        return theta_net, log_sigma_u, log_sigma_f

    def get_initial_theta(self, log_sigma_u_init=0.0, log_sigma_f_init=0.0):
        """
        Returns an initial full HMC vector, combining current network weights
        with initial guesses for log_sigma_u and log_sigma_f.

        Args:
            log_sigma_u_init: initial log(sigma_u). Default 0.0 => sigma_u = 1.0
            log_sigma_f_init: initial log(sigma_f). Default 0.0 => sigma_f = 1.0
                              Set to log(true_sigma) if you want to start near the truth.
        """
        theta_net = parameters_to_vector(self.parameters()).detach()
        log_sigmas = torch.tensor([log_sigma_u_init, log_sigma_f_init], dtype=torch.float32)
        return torch.cat([theta_net, log_sigmas])

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
        """
        Half-Normal prior on sigma, expressed in log-space.

        If sigma ~ HalfNormal(scale=1), then:
            log p(sigma) = log(2) - 0.5 * sigma^2 - log(scale) - 0.5*log(2*pi)
        
        With the change of variables sigma = exp(log_sigma), the Jacobian adds log_sigma:
            log p(log_sigma) = log p(sigma) + log_sigma
        
        We drop constants since HMC only needs the gradient:
            log p(log_sigma) ∝ -0.5 * exp(2 * log_sigma) + log_sigma
        """
        sigma = torch.exp(log_sigma)
        return -0.5 * sigma ** 2 + log_sigma

    # =========================================================================
    # Potential energy — the core of the extension
    # =========================================================================
    def potential_energy(self, theta_full, x_u, y_u, x_f, y_f, pde_problem):
        """
        Extended potential energy U(theta_full) where theta_full includes
        network weights AND log_sigma_u, log_sigma_f.

        U = - [log p(D_u | theta, sigma_u)
              + log p(D_f | theta, sigma_f)
              + log p(theta)
              + log p(sigma_u)
              + log p(sigma_f)]

        Note: compared to the original BNN, sigma_u and sigma_f are NO LONGER
        passed as fixed arguments — they are inferred from theta_full.
        """
        # 1. Split the augmented vector
        theta_net, log_sigma_u, log_sigma_f = self.split_theta(theta_full)

        # 2. Recover sigma values (always positive via exp)
        sigma_u = torch.exp(log_sigma_u)
        sigma_f = torch.exp(log_sigma_f)

        # 3. Data likelihood: p(D_u | theta, sigma_u)
        u_pred = self.functional_forward(theta_net, x_u)
        # Gaussian log-likelihood includes the log(sigma) normalization term
        # -0.5 * N * log(2*pi*sigma^2) - 0.5 * sum((pred - obs)^2 / sigma^2)
        # The constant -0.5*N*log(2*pi) is dropped (doesn't affect HMC gradients)
        N_u = y_u.shape[0]
        log_lik_u = (
            -N_u * log_sigma_u
            - 0.5 * torch.sum((u_pred - y_u) ** 2) / (sigma_u ** 2)
        )

        # 4. Physics likelihood: p(D_f | theta, sigma_f)
        x_f.requires_grad_(True)

        def u_func_for_pde(x):
            return self.functional_forward(theta_net, x)

        res_f = pde_problem.compute_residual(u_func_for_pde, x_f)
        N_f = res_f.shape[0]
        log_lik_f = (
            -N_f * log_sigma_f
            - 0.5 * torch.sum(res_f ** 2) / (sigma_f ** 2)
        )

        # 5. Prior on network weights
        log_p_theta = self.log_prior_theta(theta_net)

        # 6. Prior on noise levels (Half-Normal in log-space)
        log_p_sigma_u = self.log_prior_sigma(log_sigma_u)
        log_p_sigma_f = self.log_prior_sigma(log_sigma_f)

        # 7. Total potential energy (negative log posterior)
        log_posterior = (
            log_lik_u + log_lik_f
            + log_p_theta
            + log_p_sigma_u + log_p_sigma_f
        )
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
        """
        Given posterior samples of shape (total_params, M),
        returns sigma_u and sigma_f samples.

        Args:
            samples: tensor of shape (total_params, M)
        Returns:
            sigma_u_samples: (M,)
            sigma_f_samples: (M,)
        """
        log_sigma_u_samples = samples[self.num_params, :]
        log_sigma_f_samples = samples[self.num_params + 1, :]
        return torch.exp(log_sigma_u_samples), torch.exp(log_sigma_f_samples)
