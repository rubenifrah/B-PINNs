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
        theta_i = states[:, k-1]
        r_i = r_tk_1.clone()
        for i in range(L):
            # leapfrog steps to update the momentum and position
            # Passes **kwargs to model.gradient so it can evaluate potential energy
            r_i = r_i - delta_t/2 * model.gradient(theta_i, **kwargs)
            theta_i = theta_i + delta_t * r_i
            r_i = r_i - delta_t/2 * model.gradient(theta_i, **kwargs)
        
        # Metropolis-Hastings step to accept or reject the proposal
        p = random.random()
        
        # acceptance probability: exp(H_old - H_new)
        # Passes **kwargs to model.hamiltonian for the same reason
        H_old = model.hamiltonian(states[:, k-1], r_tk_1, **kwargs)
        H_new = model.hamiltonian(theta_i, r_i, **kwargs)
        alpha = min(1, torch.exp(H_old - H_new))
        
        if p < alpha:
            # accept the proposal
            states[:, k] = theta_i
        else:
            # reject the proposal
            states[:, k] = states[:, k-1]
    # return the last M samples
    return states[:, -M:]



