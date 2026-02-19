import torch
import numpy as np
import random
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import torch.optim as optim

def HMC_sampler(model, M, N, delta_t, theta_0, L, Mass_matrix = torch.eye(1)):
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
    """
    dim_M = Mass_matrix.shape[0]
    dim_T = theta_0.shape[1]
    if dim_M != dim_T:
        raise ValueError("Mass matrix and initial states must have the same dimension")

    states = torch.zeros(dim_T, N+1) # shape (dim_T, N+1) 
    states[:, 0] = theta_0

    for k in range(1, N + 1):
        r_tk_1 = torch.randn(dim_T)
        theta_i = states[:, k-1]
        r_i = r_tk_1.clone()
        for i in range(L):
            # leapfrog steps to update the momentum and position
            r_i = r_i - delta_t/2 * model.gradient(theta_i)
            theta_i = theta_i + delta_t * r_i
            r_i = r_i - delta_t/2 * model.gradient(theta_i)
        # Metropolis-Hastings step to accept or reject the proposal
        p = random.random()
        # acceptance probability: exp(H_old - H_new)
        alpha = min(1, torch.exp(model.hamiltonian(states[:, k-1], r_tk_1) - model.hamiltonian(theta_i, r_i)))
        if p < alpha:
            # accept the proposal
            states[:, k] = theta_i
        else:
            # reject the proposal
            states[:, k] = states[:, k-1]
    # return the last M samples
    return states[:, -M:]



