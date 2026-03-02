import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F

class BNN_UnknownNoise_f(nn.Module):
    # =========================================================================
    # Extension of the original BNN for B-PINNs where the forcing term noise
    # level sigma_f is treated as an unknown parameter and inferred jointly
    # with the network weights via HMC.
    #
    # Motivation: in the original paper (Yang et al., 2020), sigma_f and sigma_b
    # are assumed known. We relax this assumption for sigma_f only, because:
    #   - sigma_f can be meaningfully inferred from N_f = 20 collocation points
    #   - sigma_b cannot be reliably inferred from only N_b = 2 boundary points
    #     (the posterior would be dominated by the prior, not the data)
    #
    # The boundary noise sigma_b is therefore kept fixed and known, exactly
    # as in the original paper.
    #
    # Implementation strategy:
    #   We augment the HMC parameter vector theta with one extra scalar: log_sigma_f.
    #   Working in log-space ensures sigma_f > 0 throughout sampling without any
    #   constrained optimization.
    #
    #   Full HMC vector:
    #       theta_full = [theta_net (num_params,), log_sigma_f (1,)]
    #
    #   This requires NO changes to the HMC sampler — it remains agnostic to
    #   the content of theta_full.
    # =========================================================================

    def __init__(self, input_dim, output_dim, hidden_dims, activation=nn.Tanh()):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.activation = activation

        # Track shapes for slicing the 1D theta vector during functional forward.
        # This is identical to the original BNN — required because we cannot use
        # vector_to_parameters inside potential_energy (it breaks the autograd graph).
        self.param_shapes = []

        self.layers = nn.ModuleList()
        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            self.param_shapes.append((h_dim, current_dim))  # Weight shape
            self.param_shapes.append((h_dim,))              # Bias shape
            current_dim = h_dim

        self.layers.append(nn.Linear(current_dim, output_dim))
        self.param_shapes.append((output_dim, current_dim))  # Weight shape
        self.param_shapes.append((output_dim,))               # Bias shape

        # num_params: number of network weights only (e.g. 481 for [20,20])
        self.num_params = sum(torch.prod(torch.tensor(s)) for s in self.param_shapes)

        # total_params: full HMC dimension = network weights + log_sigma_f
        self.total_params = self.num_params + 1

    # =========================================================================
    # Helper: split the full HMC vector into network weights and log_sigma_f
    # =========================================================================
    def split_theta(self, theta_full):
        """
        Splits the augmented HMC vector into:
            - theta_net   : network weights, shape (num_params,)
            - log_sigma_f : scalar, the log of the inferred forcing term noise
        """
        if theta_full.numel() != self.total_params:
            raise ValueError(
                f"Expected theta_full of size {self.total_params}, "
                f"got {theta_full.numel()}. "
                f"Remember: theta_full = [theta_net ({self.num_params},), log_sigma_f (1,)]"
            )
        theta_net   = theta_full[:self.num_params]
        log_sigma_f = theta_full[self.num_params]
        return theta_net, log_sigma_f

    def get_initial_theta(self, log_sigma_f_init=-2.3):
        """
        Builds the initial full HMC vector from current network weights
        and a chosen starting value for log_sigma_f.

        Args:
            log_sigma_f_init: initial log(sigma_f).
                              Default -2.3 => sigma_f = exp(-2.3) ≈ 0.1.
                              Should be set after pretraining so that
                              parameters_to_vector picks up pretrained weights.
        Returns:
            theta_full: tensor of shape (total_params,) = (num_params + 1,)
        """
        theta_net   = parameters_to_vector(self.parameters()).detach()
        log_sigma_f = torch.tensor([log_sigma_f_init], dtype=torch.float32)
        return torch.cat([theta_net, log_sigma_f])

    # =========================================================================
    # Forward passes — identical to original BNN
    # =========================================================================
    def forward(self, x):
        """Standard forward pass using internal parameters. Used during pretraining."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)

    def functional_forward(self, theta_net, x):
        """
        Forward pass using explicit network weights theta_net.
        Keeps theta_net in the autograd computation graph (no vector_to_parameters).
        Called inside potential_energy so gradients flow back to theta_full.
        """
        if theta_net.numel() != self.num_params:
            raise ValueError(
                f"Expected theta_net of size {self.num_params}, got {theta_net.numel()}"
            )

        start     = 0
        current_x = x
        num_layers = len(self.hidden_dims) + 1

        for i in range(num_layers):
            # Extract weight matrix
            w_shape = self.param_shapes[2 * i]
            w_numel = w_shape[0] * w_shape[1]
            weight  = theta_net[start:start + w_numel].view(w_shape)
            start  += w_numel

            # Extract bias vector
            b_shape = self.param_shapes[2 * i + 1]
            b_numel = b_shape[0]
            bias    = theta_net[start:start + b_numel]
            start  += b_numel

            current_x = F.linear(current_x, weight, bias)

            if i < num_layers - 1:
                current_x = self.activation(current_x)

        return current_x

    # =========================================================================
    # Priors
    # =========================================================================
    def log_prior_theta(self, theta_net, sigma_theta=1.0):
        """
        Standard Gaussian prior on network weights (identical to original BNN):
            log P(theta) = -0.5 * sum(theta^2) / sigma_theta^2
        """
        return -0.5 * torch.sum(theta_net ** 2) / (sigma_theta ** 2)

    def log_prior_sigma_f(self, log_sigma_f):
        """
        Half-Normal prior on sigma_f, expressed in log-space.

        We want: sigma_f ~ HalfNormal(scale=1), i.e. soft preference for small
        positive values while remaining weakly informative.

        Change of variables sigma_f = exp(log_sigma_f) introduces a Jacobian:
            log P(log_sigma_f) = log P(sigma_f) + log|d(sigma_f)/d(log_sigma_f)|
                               = log P(sigma_f) + log_sigma_f

        Dropping constants (they don't affect HMC gradients):
            log P(log_sigma_f) ∝ -0.5 * sigma_f^2 + log_sigma_f

        The -0.5*sigma_f^2 term penalizes large sigma_f (prior pushes toward small noise).
        The +log_sigma_f term is the Jacobian correction for the change of variables.
        Without it, the prior on sigma_f would be distorted.
        """
        sigma_f = torch.exp(log_sigma_f)
        return -0.5 * sigma_f ** 2 + log_sigma_f

    # def log_prior_sigma_f(self, log_sigma_f, alpha=2.0, beta=0.02):
    #     """
    #     Inverse-Gamma prior on sigma_f^2, expressed in log-space.

    #     sigma_f^2 ~ InvGamma(alpha, beta)

    #     With alpha=2.0, beta=0.02:
    #         Prior mode of sigma_f^2 = beta/(alpha+1) = 0.02/3 ≈ 0.0067
    #         => Prior mode of sigma_f ≈ 0.082  (close to true 0.1)

    #     In log-space with Jacobian correction:
    #         log p(log_sigma_f) ∝ -2*alpha * log_sigma_f - beta * exp(-2*log_sigma_f)
    #     """
    #     sigma_f_sq = torch.exp(2 * log_sigma_f)
    #     return -2 * alpha * log_sigma_f - beta / sigma_f_sq

    # =========================================================================
    # Potential energy U(theta_full) — core of the extension
    # =========================================================================
    def potential_energy(self, theta_full, x_b, y_b, sigma_b, x_f, y_f, pde_problem):
        """
        Extended potential energy where sigma_f is inferred and sigma_b is fixed.

        Following the paper's notation (Eq. 10):
            U(theta_full) = - log P(D_f | theta_full)
                            - log P(D_b | theta_full, sigma_b)
                            - log P(theta)
                            - log P(sigma_f)

        Args:
            theta_full  : augmented HMC vector [theta_net, log_sigma_f], shape (total_params,)
            x_b         : boundary condition locations, shape (N_b, 1)
            y_b         : noisy boundary observations b̄, shape (N_b, 1)
            sigma_b     : FIXED boundary noise level (known, as in original paper)
            x_f         : forcing term collocation points, shape (N_f, 1)
            y_f         : noisy forcing term measurements f̄, shape (N_f, 1)
            pde_problem : PDE residual evaluator (e.g. Poisson1D instance)
        """
        # 1. Split theta_full into network weights and log_sigma_f
        theta_net, log_sigma_f = self.split_theta(theta_full)

        # 2. Recover sigma_f (always positive via exp, differentiable w.r.t. theta_full)
        sigma_f = torch.exp(log_sigma_f)

        # 3. Boundary condition likelihood: P(D_b | theta, sigma_b)
        #    sigma_b is fixed — normalization term is constant, dropped as in original paper.
        #    Using paper notation: b̃ = B_x(ũ; lambda), compared to b̄ measurements.
        b_pred   = self.functional_forward(theta_net, x_b)
        log_lik_b = -0.5 * torch.sum((b_pred - y_b) ** 2) / (sigma_b ** 2)

        # 4. Forcing term likelihood: P(D_f | theta, sigma_f)
        #    sigma_f is INFERRED — normalization term -N_f * log_sigma_f MUST be included.
        #    It is what creates the self-calibration: large sigma_f reduces the residual
        #    penalty but incurs cost -N_f * log_sigma_f. The posterior balances these.
        x_f.requires_grad_(True)

        def u_func_for_pde(x):
            return self.functional_forward(theta_net, x)

        res_f    = pde_problem.compute_residual(u_func_for_pde, x_f)
        N_f      = res_f.shape[0]
        log_lik_f = (
            - N_f * log_sigma_f
            - 0.5 * torch.sum(res_f ** 2) / (sigma_f ** 2)
        )

        # 5. Prior on network weights (identical to original BNN)
        log_p_theta = self.log_prior_theta(theta_net)

        # 6. Prior on sigma_f (Half-Normal in log-space, with Jacobian correction)
        log_p_sigma_f = self.log_prior_sigma_f(log_sigma_f)

        # 7. Total potential energy = negative log posterior
        #    U = - [log P(D_b|.) + log P(D_f|.) + log P(theta) + log P(sigma_f)]
        log_posterior = log_lik_b + log_lik_f + log_p_theta + log_p_sigma_f
        return -log_posterior

    # =========================================================================
    # Hamiltonian and gradient — called by HMC sampler, unchanged in structure
    # =========================================================================
    def hamiltonian(self, theta_full, r, **kwargs):
        """
        H(theta_full, r) = U(theta_full) + K(r)
        K(r) = 0.5 * r^T r  (identity mass matrix, as in original paper)
        The HMC sampler calls this — signature unchanged from original BNN.
        """
        U = self.potential_energy(theta_full, **kwargs)
        K = 0.5 * torch.sum(r ** 2)
        return U + K

    def gradient(self, theta_full, **kwargs):
        """
        Gradient of U w.r.t. theta_full (all num_params + 1 dimensions).
        The last dimension gives dU/d(log_sigma_f), which HMC uses to update
        the sigma_f estimate alongside the network weights.
        Called by HMC sampler — signature unchanged from original BNN.
        """
        theta_copy = theta_full.clone().detach().requires_grad_(True)
        U          = self.potential_energy(theta_copy, **kwargs)
        grad       = torch.autograd.grad(U, theta_copy)[0]
        return grad

    # =========================================================================
    # Utility: extract inferred sigma_f posterior from HMC samples
    # =========================================================================
    def extract_sigma_f_samples(self, samples):
        """
        Extracts posterior samples of sigma_f from the full HMC sample matrix.

        Args:
            samples: tensor of shape (total_params, M) from HMC_sampler
        Returns:
            sigma_f_samples: tensor of shape (M,) — posterior samples of sigma_f
        """
        log_sigma_f_samples = samples[self.num_params, :]   # last row
        return torch.exp(log_sigma_f_samples)
