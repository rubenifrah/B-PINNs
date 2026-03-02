import torch


class PDEProblem:
    """
    Base class for all PDE problems.
    """
    def __init__(self, x_f, y_f, sigma_f):
        self.x_f     = x_f.detach().clone().requires_grad_(True)
        self.y_f     = y_f
        self.sigma_f = sigma_f

    def compute_residual(self, u_func, x):
        raise NotImplementedError


class Poisson1D(PDEProblem):
    """
    1D Poisson equation: u_xx = f(x)
    Forward problem, Section 3.2.1.
    """
    def __init__(self, x_f, y_f, sigma_f):
        super().__init__(x_f, y_f, sigma_f)

    def compute_residual(self, u_func, x, params=None):
        u    = u_func(x)
        u_x  = torch.autograd.grad(u,   x,   grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        return u_xx - self.y_f


class Burgers1D(PDEProblem):
    """
    1D Burgers equation: u_t + u * u_x = nu * u_xx
    """
    def compute_residual(self, u_func, xt, params=None):
        xt.requires_grad_(True)
        u     = u_func(xt)
        grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
        u_t   = grads[:, 0:1]
        u_x   = grads[:, 1:2]
        u_xx  = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x),
                                    create_graph=True)[0][:, 1:2]
        nu    = 0.01 / torch.pi
        return u_t + u * u_x - nu * u_xx


class InverseReactionDiffusion1D(PDEProblem):
    """
    1D nonlinear diffusion-reaction equation — INVERSE problem, Section 3.3.1.

        lambda * u_xx + k * tanh(u) = f,    x in [-0.7, 0.7]

    Parameters:
        lambda = 0.01  KNOWN  (fixed diffusion coefficient)
        k              UNKNOWN — inferred jointly with BNN weights via HMC

    The unknown k is passed via `params` from BNN_Inverse.potential_energy(),
    which extracts it from theta_full = [theta_net, k].

    True solution: u = sin^3(6x),  true k = 0.7
    """
    def __init__(self, x_f, y_f, sigma_f, lambda_val=0.01):
        super().__init__(x_f, y_f, sigma_f)
        self.lambda_val = lambda_val

    def compute_residual(self, u_func, x, params):
        """
        Args:
            u_func : BNN functional forward for current theta_net
            x      : collocation points, shape (N_f, 1), requires_grad=True
            params : tensor of shape (1,) containing current HMC sample for k
        """
        k    = params[0]
        u    = u_func(x)
        u_x  = torch.autograd.grad(u,   x,   grad_outputs=torch.ones_like(u),
                                   create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x,   grad_outputs=torch.ones_like(u_x),
                                   create_graph=True)[0]
        # Residual: lambda * u_xx + k * tanh(u) - f = 0
        return self.lambda_val * u_xx + k * torch.tanh(u) - self.y_f
