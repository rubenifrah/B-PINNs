"""
NNGP kernels for the infinite-width limit of Bayesian Neural Networks.

Implements the exact recursive covariance formula for fully connected networks
with various activation functions. These are used as the GP prior kernel in
the Physics-Informed GP framework.

References:
  - Lee et al., "Deep Neural Networks as Gaussian Processes" (ICLR 2018)
  - Matthews et al., "Gaussian Process Behaviour in Wide Deep Neural Networks" (2018)
"""
import numpy as np
from scipy import integrate


def _expect_tanh_tanh(K, K_xx, K_xpxp):
    """
    Compute E[tanh(z1) * tanh(z2)] where (z1, z2) ~ N(0, Sigma)
    with Sigma = [[K_xx, K], [K, K_xpxp]].
    
    Uses numerical integration (Gauss-Hermite quadrature) for the tanh
    activation since there is no known closed-form expression.
    """
    # Correlation coefficient
    std_x = np.sqrt(np.maximum(K_xx, 1e-12))
    std_xp = np.sqrt(np.maximum(K_xpxp, 1e-12))
    rho = np.clip(K / (std_x * std_xp + 1e-12), -1 + 1e-10, 1 - 1e-10)
    
    # Use Gauss-Hermite quadrature for the 2D integral
    # E[tanh(z1)tanh(z2)] where z1, z2 are correlated Gaussians
    n_quad = 50
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    
    # Transform: z1 = std_x * sqrt(2) * t1
    #            z2 = std_xp * sqrt(2) * (rho * t1 + sqrt(1-rho^2) * t2)
    result = 0.0
    sqrt_1_minus_rho2 = np.sqrt(np.maximum(1.0 - rho**2, 1e-12))
    
    for i in range(n_quad):
        z1 = std_x * np.sqrt(2.0) * nodes[i]
        tanh_z1 = np.tanh(z1)
        for j in range(n_quad):
            z2 = std_xp * np.sqrt(2.0) * (rho * nodes[i] + sqrt_1_minus_rho2 * nodes[j])
            tanh_z2 = np.tanh(z2)
            result += weights[i] * weights[j] * tanh_z1 * tanh_z2
    
    result /= np.pi  # Gauss-Hermite normalization for 2D
    return result


def tanh_nngp_kernel(x, xp, depth, sigma_w2=1.0, sigma_b2=1.0):
    """
    Compute the NNGP kernel for a fully connected network with tanh activation.
    
    The kernel is defined recursively through the layers:
      K^{(0)}(x, x') = σ_w² (x · x') + σ_b²
      K^{(l+1)}(x, x') = σ_w² E[tanh(z1)tanh(z2)] + σ_b²
    where (z1, z2) ~ N(0, [[K^(l)(x,x), K^(l)(x,x')], [K^(l)(x,x'), K^(l)(x',x')]])
    
    Args:
        x, xp: Scalar inputs (float64).
        depth: Number of hidden layers.
        sigma_w2: Weight variance (σ_w²).
        sigma_b2: Bias variance (σ_b²).
    
    Returns:
        Scalar float64 value of the kernel.
    """
    x = float(x)
    xp = float(xp)
    
    # Layer 0: linear kernel
    K = sigma_w2 * (x * xp) + sigma_b2
    K_xx = sigma_w2 * (x * x) + sigma_b2
    K_xpxp = sigma_w2 * (xp * xp) + sigma_b2
    
    # Recursive layers
    for _ in range(depth):
        E_val = _expect_tanh_tanh(K, K_xx, K_xpxp)
        E_xx = _expect_tanh_tanh(K_xx, K_xx, K_xx)
        E_xpxp = _expect_tanh_tanh(K_xpxp, K_xpxp, K_xpxp)
        
        K = sigma_w2 * E_val + sigma_b2
        K_xx = sigma_w2 * E_xx + sigma_b2
        K_xpxp = sigma_w2 * E_xpxp + sigma_b2
    
    return K


def nngp_kernel_matrix(X1, X2, depth, sigma_w2=1.0, sigma_b2=1.0):
    """
    Compute the full kernel matrix K[i,j] = k(X1[i], X2[j]).
    
    Args:
        X1: (N,) array of input points.
        X2: (M,) array of input points.
        depth: Number of hidden layers.
    
    Returns:
        (N, M) kernel matrix.
    """
    N = len(X1)
    M = len(X2)
    K = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            K[i, j] = tanh_nngp_kernel(X1[i], X2[j], depth, sigma_w2, sigma_b2)
    return K
