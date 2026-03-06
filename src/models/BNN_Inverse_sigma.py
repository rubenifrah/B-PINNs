import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F


class BNN_Inverse(nn.Module):
    """
    B-PINN for the inverse problem of Section 3.3.1 (Yang et al., 2020),
    extended to also infer the interior measurement noise sigma_u.

    PDE: lambda * u_xx + k * tanh(u) = f,   x in [-0.7, 0.7]

    Two unknowns are inferred jointly with the BNN weights theta via HMC:
        - k        : unknown PDE reaction coefficient (true value: 0.7)
        - sigma_u  : unknown noise level on interior u measurements

    Extended joint posterior:
        P(theta, k, sigma_u | D) proportional to
            P(D_u | theta, sigma_u) * P(D_f | theta, k) * P(D_b | theta)
            * P(theta) * P(k) * P(sigma_u)

    Augmented HMC vector:
        theta_full = [theta_net (num_params,), k (1,), log_sigma_u (1,)]

    sigma_u is sampled in log-space (phi_u = log sigma_u) to enforce positivity.
    k is sampled directly (real-valued, no constraint needed).
    sigma_b and sigma_f remain FIXED and KNOWN as in the original paper.

    Dataset (Section 3.3.1):
        D_u : 6 interior sensors  — direct observations of u(x)  [sigma_u INFERRED]
        D_b : 2 boundary sensors  — direct observations of u      [sigma_b FIXED]
        D_f : 32 sensors          — noisy measurements of f(x)   [sigma_f FIXED]

    Why sigma_u is better motivated than sigma_f in the forward problem:
        sigma_u appears in P(D_u | theta, sigma_u) which compares ũ(x_u) directly
        to noisy observations ū. The network's freedom to adjust R_u(theta) is
        limited by the simultaneous PDE constraint (D_f) and boundary constraint (D_b),
        making sigma_u more genuinely identifiable than sigma_f was in the forward problem,
        where the PDE residual and sigma_f were fully entangled.
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

        # Number of network weights only
        self.num_params   = sum(torch.prod(torch.tensor(s)) for s in self.param_shapes)

        # Total HMC dimension: network weights + k + log_sigma_u
        self.total_params = self.num_params + 2

        # Indices for clarity
        self.idx_k           = self.num_params        # position of k in theta_full
        self.idx_log_sigma_u = self.num_params + 1    # position of log_sigma_u

    # =========================================================================
    # Split / build theta_full
    # =========================================================================
    def split_theta(self, theta_full):
        """
        Splits theta_full into:
            theta_net   : network weights,          shape (num_params,)
            k           : unknown PDE coefficient,  shape (1,)
            log_sigma_u : log of interior u noise,  scalar
        """
        if theta_full.numel() != self.total_params:
            raise ValueError(
                f"Expected theta_full size {self.total_params}, "
                f"got {theta_full.numel()}. "
                f"theta_full = [theta_net ({self.num_params},), k (1,), log_sigma_u (1,)]"
            )
        theta_net   = theta_full[:self.num_params]
        k           = theta_full[self.idx_k:self.idx_k + 1]
        log_sigma_u = theta_full[self.idx_log_sigma_u]
        return theta_net, k, log_sigma_u

    def get_initial_theta(self, k_init=0.5, log_sigma_u_init=0.0):
        """
        Builds the initial HMC vector from pretrained weights + initial k + initial log_sigma_u.

        Args:
            k_init           : initial guess for k. Default 0.5 (true: 0.7).
                               Set deliberately away from truth to test inference.
            log_sigma_u_init : initial log(sigma_u). Default 0.0 => sigma_u = 1.0.
                               Use log(true_sigma_u) to start near truth,
                               or a different value to test convergence.
        """
        theta_net   = parameters_to_vector(self.parameters()).detach()
        k           = torch.tensor([k_init],           dtype=torch.float32)
        log_sigma_u = torch.tensor([log_sigma_u_init], dtype=torch.float32)
        return torch.cat([theta_net, k, log_sigma_u])

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
        """Gaussian prior on network weights: log P(theta) = -0.5*||theta||^2"""
        return -0.5 * torch.sum(theta_net ** 2) / (sigma_theta ** 2)

    def log_prior_k(self, k, sigma_k=1.0):
        """
        Gaussian prior on k: log P(k) = -0.5 * k^2 / sigma_k^2
        Weakly informative — prevents k from drifting to extreme values.
        """
        return -0.5 * torch.sum(k ** 2) / (sigma_k ** 2)

    def log_prior_sigma_u(self, log_sigma):
        mu_log_sigma = 0  # log(0.1)
        tau = 0.5            # flexibility 
        return -0.5 * ((log_sigma - mu_log_sigma) / tau) ** 2

    # =========================================================================
    # Potential energy
    # =========================================================================
    def potential_energy(self, theta_full, x_u, y_u, x_b, y_b,
                         x_f, y_f, sigma_b, sigma_f, pde_problem):
        """
        Extended potential energy with sigma_u inferred:

        U(theta_full) = - log P(D_u | theta, sigma_u)    [sigma_u INFERRED]
                        - log P(D_b | theta)              [sigma_b FIXED]
                        - log P(D_f | theta, k)           [sigma_f FIXED, k INFERRED]
                        - log P(theta)
                        - log P(k)
                        - log P(sigma_u)

        Args:
            theta_full  : [theta_net, k, log_sigma_u], shape (total_params,)
            x_u, y_u   : 6 interior u observations
            x_b, y_b   : 2 boundary observations
            x_f, y_f   : 32 forcing term observations
            sigma_b    : FIXED known boundary noise
            sigma_f    : FIXED known forcing term noise
            pde_problem: InverseReactionDiffusion1D instance

        Note: sigma_u is intentionally NOT an argument — it is inferred from theta_full.
        """
        theta_net, k, log_sigma_u = self.split_theta(theta_full)

        # Recover sigma_u from log-space (always positive, differentiable w.r.t. theta_full)
        sigma_u = torch.exp(log_sigma_u)

        # 1. Interior u likelihood: P(D_u | theta, sigma_u)
        #    sigma_u is INFERRED — normalization term -N_u*log_sigma_u must be included.
        #    It prevents the sampler from trivially setting sigma_u -> infinity.
        #    The tension: -N_u*log_sigma_u penalizes large sigma_u,
        #                 R_u/2*sigma_u^2 penalizes small sigma_u.
        #    The posterior finds the sigma_u that best explains the 6 observations.
        u_pred    = self.functional_forward(theta_net, x_u)
        N_u       = y_u.shape[0]
        log_lik_u = (
            - N_u * log_sigma_u
            - 0.5 * torch.sum((u_pred - y_u) ** 2) / (sigma_u ** 2)
        )

        # 2. Boundary likelihood: P(D_b | theta)
        #    sigma_b is FIXED — normalization is constant, dropped as in original paper
        b_pred    = self.functional_forward(theta_net, x_b)
        log_lik_b = -0.5 * torch.sum((b_pred - y_b) ** 2) / (sigma_b ** 2)

        # 3. Forcing term likelihood: P(D_f | theta, k)
        #    sigma_f is FIXED — k is passed via params to compute_residual
        x_f.requires_grad_(True)

        def u_func_for_pde(x):
            return self.functional_forward(theta_net, x)

        res_f     = pde_problem.compute_residual(u_func_for_pde, x_f, params=k)
        log_lik_f = -0.5 * torch.sum(res_f ** 2) / (sigma_f ** 2)

        # 4. Priors
        log_p_theta   = self.log_prior_theta(theta_net)
        log_p_k       = self.log_prior_k(k)
        log_p_sigma_u = self.log_prior_sigma_u(log_sigma_u)

        # 5. Total potential energy = negative log posterior
        log_posterior = (
            log_lik_u + log_lik_b + log_lik_f
            + log_p_theta + log_p_k + log_p_sigma_u
        )
        return -log_posterior

    # =========================================================================
    # Hamiltonian and gradient — called by HMC sampler, unchanged in structure
    # =========================================================================
    def hamiltonian(self, theta_full, r, **kwargs):
        U = self.potential_energy(theta_full, **kwargs)
        K = 0.5 * torch.sum(r ** 2)
        return U + K

    def gradient(self, theta_full, **kwargs):
        """
        Gradient of U w.r.t. theta_full (num_params + 2 dimensions).
        The last two dimensions give dU/dk and dU/d(log_sigma_u).
        HMC uses these to update k and sigma_u alongside network weights.
        """
        theta_copy = theta_full.clone().detach().requires_grad_(True)
        U          = self.potential_energy(theta_copy, **kwargs)
        grad       = torch.autograd.grad(U, theta_copy)[0]
        return grad

    # =========================================================================
    # Utilities
    # =========================================================================
    def extract_k_samples(self, samples):
        """Returns posterior samples of k, shape (M,)"""
        return samples[self.idx_k, :]

    def extract_sigma_u_samples(self, samples):
        """Returns posterior samples of sigma_u, shape (M,)"""
        return torch.exp(samples[self.idx_log_sigma_u, :])

    def predict_f(self, theta_net, x, k_val, lambda_val):
        """
        Computes f̃(x) = lambda*u_xx + k*tanh(u) for a single posterior sample.
        Used for plotting the posterior predictive over f(x).
        """
        x_in = x.clone().detach().requires_grad_(True)
        u    = self.functional_forward(theta_net, x_in)
        u_x  = torch.autograd.grad(
            u, x_in, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(
            u_x, x_in, grad_outputs=torch.ones_like(u_x), create_graph=False)[0]
        f_pred = lambda_val * u_xx + k_val * torch.tanh(u.detach())
        return f_pred.detach()