import torch
import numpy as np
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

# =========================================================================
# The Bayesian Neural Network (BNN) needs data (x_u, y_u), physics points (x_f),
# and the PDE problem definition to evaluate the potential energy U(theta).
# Using `**kwargs` and passing them to `model.gradient` and `model.hamiltonian`
# allows the B-PINN to dynamically evaluate the posterior based on the current dataset.
# =========================================================================
def HMC_sampler(model, M, N, delta_t, theta_0, L, Mass_matrix=torch.eye(1), **kwargs):
    """
        HMC sampler for the B-PINNs

        Args:
            model: The model to sample from
            M: Number of samples
            N: Number of total steps
            delta_t: Step size
            theta_0: Initial states of the parameters
            L: Number of leapfrog steps
            Mass_matrix: Mass matrix (assumed to be identity for simplicity)
            **kwargs: Additional context passed to the model (e.g., data, pde_problem) needed for 
                evaluating Hamiltonian and gradient.
    """
    dim_M = Mass_matrix.shape[0]
    dim_T = theta_0.shape[0]  # Get length of 1D vector theta_0
    if dim_M != dim_T and dim_M != 1:
        # Note: If Mass matrix is I(1), it relies on broadcasting, but standard 
        # HMC usually needs M to match T or be purely scalar. Added check guard.
        print("Warning: Mass matrix dimension does not match the number of parameters.")

    states = torch.zeros(dim_T, N+1) # shape (dim_T, N+1) 
    states[:, 0] = theta_0

    for k in range(1, N + 1):
        r_tk_1 = torch.randn(dim_T)
        
        # We must clone AND detach to prevent PyTorch from building 
        # an infinitely long computation graph across HMC steps!
        theta_i = states[:, k-1].clone().detach()
        r_i = r_tk_1.clone().detach()
        
        for i in range(L):
            # 1. Half step for momentum
            # We get the gradient. The gradient function automatically handles its own internal graph.
            grad_U_half = model.gradient(theta_i, **kwargs)
            # Clip gradients to prevent explosion during leapfrog
            grad_U_half = torch.clamp(grad_U_half, -10.0, 10.0)
            
            with torch.no_grad():
                r_i = r_i - (delta_t / 2.0) * grad_U_half
                
                # 2. Full step for position
                theta_i = theta_i + delta_t * r_i
                
            # 3. Half step for momentum
            grad_U_full = model.gradient(theta_i, **kwargs)
            grad_U_full = torch.clamp(grad_U_full, -10.0, 10.0)
            
            with torch.no_grad():
                r_i = r_i - (delta_t / 2.0) * grad_U_full
        
        # Metropolis-Hastings step to accept or reject the proposal
        p = random.random()
        
        # evaluate energies (we must NOT use no_grad because computing U(theta) 
        # requires autograd to compute the PDE spatial derivatives!)
        H_old = model.hamiltonian(states[:, k-1], r_tk_1, **kwargs)
        H_new = model.hamiltonian(theta_i, r_i, **kwargs)
        
        # safeguard exponent against overflow and drop graph tracking via .item()
        H_diff = torch.clamp(H_old - H_new, max=20.0, min=-50.0).item()
        alpha = min(1.0, math.exp(H_diff))
        
        if p < alpha:
            # accept the proposal
            states[:, k] = theta_i
        else:
            # reject the proposal
            states[:, k] = states[:, k-1]
            
    # return the last M samples
    return states[:, -M:]



