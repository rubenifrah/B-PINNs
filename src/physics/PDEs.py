import torch

class PDEProblem:
    """
    Base class for all PDE problems.
    To properly define a new PDE problem, inherit from this class and override the compute_residual method.
    """
    def __init__(self, x_f, y_f, sigma_f):
        self.x_f = x_f.detach().clone().requires_grad_(True)
        self.y_f = y_f
        self.sigma_f = sigma_f

    def compute_residual(self, u_func, x):
        """This must be overridden by specific PDEs"""
        raise NotImplementedError

class Poisson1D(PDEProblem):
    """
    1D Poisson equation: u_xx = f(x)
    """
    def __init__(self, x_f, y_f, sigma_f):
        super().__init__(x_f, y_f, sigma_f)

    def compute_residual(self, u_func, x, params=None):
        u = u_func(x)
        # First derivative
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        # Second derivative
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_xx - self.y_f # residual: u_xx - f = 0

class Burgers1D(PDEProblem):
    """
    1D Burgers equation: u_t + u * u_x = nu * u_xx
    """
    def compute_residual(self, u_func, xt, params=None):
        # xt is a tensor of shape (N, 2) where column 0 is t and column 1 is x
        xt.requires_grad_(True)
        u = u_func(xt)
        
        # Gradients w.r.t input
        grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grads[:, 0:1]
        u_x = grads[:, 1:2]
        
        # Second derivative u_xx
        u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        
        nu = 0.01 / torch.pi
        return u_t + u * u_x - nu * u_xx
    
class InverseReactionDiffusion1D(PDEProblem):
    """
    In inverse problems, the physics parameter lambda is unknown and part of the Bayesian posterior
    This example demonstrates how to handle inverse problems in B-PINNs, for potential future extansion of the project
    """
    def compute_residual(self, u_func, x, params):
        """
        params: a tensor containing the current HMC sample for lambda
        """
        u = u_func(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # The unknown parameter lambda is passed from the HMC sampler
        lambda_val = params[0] 
        
        # Residual: u_xx - lambda * u - f = 0
        return u_xx - lambda_val * u - self.y_f