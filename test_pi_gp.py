"""
Test the Analytical NNGP PI-GP on the 1D Poisson problem.
Matches the paper's experimental setup exactly.

Paper setup:
  PDE:          λ ∂²_x u = f,  x ∈ [-0.7, 0.7],  λ = 0.01
  True solution: u(x) = sin³(6x)
  Architecture:  2 hidden layers, 50 neurons each, tanh activation
  f sensors:     16 equidistant in [-0.7, 0.7]
  u sensors:     2 boundary at x = ±0.7
  Case 1:        ε_f ~ N(0, 0.01²),  ε_b ~ N(0, 0.01²)
  Case 2:        ε_f ~ N(0, 0.1²),   ε_b ~ N(0, 0.1²)
"""
import numpy as np
import matplotlib.pyplot as plt

from src.models.PIGP import AnalyticalPIGP

np.random.seed(42)

# ---- Paper experimental parameters ----
lam = 0.01

# Boundary sensors
x_b = np.array([-0.7, 0.7])
u_true_b = np.sin(6 * x_b)**3

# Physics sensors: 16 equidistant
x_f = np.linspace(-0.7, 0.7, 16)

# True solution
def u_exact(x): return np.sin(6 * x)**3
def u_xx_exact(x):
    s, c = np.sin(6*x), np.cos(6*x)
    return 108 * s * (3 * c**2 - 1)
def f_exact(x): return lam * u_xx_exact(x)

y_b_true = u_true_b
y_f_true = f_exact(x_f)

# Test grid
x_test = np.linspace(-0.8, 0.8, 100)

# ---- Run both noise cases ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

for case_idx, (sigma_b, sigma_f, title) in enumerate([
    (0.01, 0.01, r"Case 1: $\sigma_f = 0.01$, $\sigma_b = 0.01$"),
    (0.1,  0.1,  r"Case 2: $\sigma_f = 0.1$, $\sigma_b = 0.1$"),
]):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    
    y_b_noisy = y_b_true + np.random.randn(len(x_b)) * sigma_b
    y_f_noisy = y_f_true + np.random.randn(len(x_f)) * sigma_f

    # PI-GP matching paper architecture: 2 hidden layers → depth=2
    pigp = AnalyticalPIGP(
        depth=2,
        sigma_w2=1.0,
        sigma_b2=1.0,
        lambda_pde=lam,
        noise_u=sigma_b,
        noise_f=sigma_f,
        fd_step=5e-3
    )

    mu, var = pigp.fit_and_predict(x_b, y_b_noisy, x_f, y_f_noisy, x_test)
    std = np.sqrt(np.maximum(var, 0))

    # Plot
    ax = axes[case_idx]
    u_true = u_exact(x_test)
    ax.fill_between(x_test, mu - 2*std, mu + 2*std, color='blue', alpha=0.15, label='±2σ GP')
    ax.fill_between(x_test, mu - std, mu + std, color='blue', alpha=0.3, label='±1σ GP')
    ax.plot(x_test, mu, 'b-', lw=2, label='GP Mean')
    ax.plot(x_test, u_true, 'k--', lw=2, label=r'True $u(x) = \sin^3(6x)$')
    ax.scatter(x_b, y_b_noisy, c='red', s=80, zorder=5, label='Boundary $u$')
    ax.scatter(x_f, y_f_noisy, c='green', s=30, zorder=4, alpha=0.7, marker='^', label='Physics $f$')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.suptitle('Analytical PI-GP (Tanh NNGP Kernel, 2-layer depth)', fontsize=14)
plt.tight_layout()
plt.savefig('test_gp_result.png', dpi=150)
print("\nSaved test_gp_result.png")
