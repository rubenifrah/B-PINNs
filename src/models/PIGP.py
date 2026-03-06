"""
Analytical Physics-Informed Gaussian Process Regressor using the NNGP kernel.

Computes kernel derivatives via central finite differences in float64.
This stays fully analytical (no Monte Carlo) while preserving the BNN→GP link.

For the 1D Poisson equation: λ u_xx(x) = f(x)
  Operator: L = λ d²/dx²
  k_uf(x, x') = λ · ∂²k/∂x'²
  k_fu(x, x') = λ · ∂²k/∂x²
  k_ff(x, x') = λ² · ∂⁴k/∂x²∂x'²
"""
import numpy as np
from src.models.NNGP import tanh_nngp_kernel


class AnalyticalPIGP:
    """
    Exact PI-GP using the analytical NNGP kernel for tanh networks.
    All kernel derivatives computed via central finite differences in float64.
    """
    def __init__(self, depth=2, sigma_w2=1.0, sigma_b2=1.0,
                 lambda_pde=0.01, noise_u=0.01, noise_f=0.01,
                 fd_step=5e-3):
        self.depth = depth
        self.sigma_w2 = sigma_w2
        self.sigma_b2 = sigma_b2
        self.lam = lambda_pde
        self.noise_u = noise_u
        self.noise_f = noise_f
        self.h = fd_step  # FD step size

    def _k(self, x, xp):
        """Evaluate the NNGP kernel at scalar points."""
        return tanh_nngp_kernel(x, xp, self.depth, self.sigma_w2, self.sigma_b2)

    # ---- Central finite difference derivatives ----
    
    def _d2k_dxp2(self, x, xp):
        """∂²k/∂x'² via 2nd-order central finite difference."""
        h = self.h
        return (self._k(x, xp + h) - 2.0 * self._k(x, xp) + self._k(x, xp - h)) / (h * h)

    def _d2k_dx2(self, x, xp):
        """∂²k/∂x² via 2nd-order central finite difference."""
        h = self.h
        return (self._k(x + h, xp) - 2.0 * self._k(x, xp) + self._k(x - h, xp)) / (h * h)

    def _d4k_dx2dxp2(self, x, xp):
        """∂⁴k/∂x²∂x'² via nested 2nd-order central finite differences."""
        h = self.h
        # Apply ∂²/∂x² to ∂²k/∂x'²
        return (self._d2k_dxp2(x + h, xp) - 2.0 * self._d2k_dxp2(x, xp) + self._d2k_dxp2(x - h, xp)) / (h * h)

    # ---- Build all block matrices ----

    def _build_blocks(self, X_u, X_f):
        """Build K_uu, K_uf, K_fu, K_ff block matrices."""
        Nu = len(X_u)
        Nf = len(X_f)
        lam = self.lam
        
        print(f"  Building K_uu ({Nu}×{Nu})...")
        K_uu = np.zeros((Nu, Nu))
        for i in range(Nu):
            for j in range(i, Nu):
                val = self._k(X_u[i], X_u[j])
                K_uu[i, j] = val
                K_uu[j, i] = val
        
        print(f"  Building K_uf ({Nu}×{Nf})...")
        K_uf = np.zeros((Nu, Nf))
        for i in range(Nu):
            for j in range(Nf):
                K_uf[i, j] = lam * self._d2k_dxp2(X_u[i], X_f[j])
        
        print(f"  Building K_fu ({Nf}×{Nu})...")
        K_fu = np.zeros((Nf, Nu))
        for i in range(Nf):
            for j in range(Nu):
                K_fu[i, j] = lam * self._d2k_dx2(X_f[i], X_u[j])
        
        print(f"  Building K_ff ({Nf}×{Nf})... (this takes a moment)")
        K_ff = np.zeros((Nf, Nf))
        for i in range(Nf):
            for j in range(i, Nf):
                val = (lam**2) * self._d4k_dx2dxp2(X_f[i], X_f[j])
                K_ff[i, j] = val
                K_ff[j, i] = val
        
        return K_uu, K_uf, K_fu, K_ff

    def fit_and_predict(self, X_u, y_u, X_f, y_f, X_test):
        """
        Compute the GP posterior predictive mean and variance.
        
        Args:
            X_u: (N_u,) boundary locations
            y_u: (N_u,) boundary observations 
            X_f: (N_f,) physics sensor locations
            y_f: (N_f,) physics observations (f = λ u_xx)
            X_test: (N_t,) test grid locations
        
        Returns:
            mu: (N_t,) posterior mean
            var: (N_t,) posterior marginal variance
        """
        X_u = np.asarray(X_u, dtype=np.float64).flatten()
        X_f = np.asarray(X_f, dtype=np.float64).flatten()
        X_test = np.asarray(X_test, dtype=np.float64).flatten()
        y_u = np.asarray(y_u, dtype=np.float64).flatten()
        y_f = np.asarray(y_f, dtype=np.float64).flatten()
        
        Nu = len(X_u)
        Nf = len(X_f)
        Nt = len(X_test)
        lam = self.lam

        # 1. Build observation block matrix
        K_uu, K_uf, K_fu, K_ff = self._build_blocks(X_u, X_f)
        
        K = np.block([[K_uu, K_uf], [K_fu, K_ff]])
        Sigma = np.diag(np.concatenate([
            np.full(Nu, self.noise_u**2),
            np.full(Nf, self.noise_f**2)
        ]))
        K_noisy = K + Sigma
        
        # Ensure PD (FD noise)
        eigvals = np.linalg.eigvalsh(K_noisy)
        min_eig = eigvals.min()
        print(f"  Min eigenvalue of K_noisy: {min_eig:.6e}")
        if min_eig < 0:
            K_noisy += (abs(min_eig) + 1e-8) * np.eye(Nu + Nf)

        # 2. Stack observations
        Y = np.concatenate([y_u, y_f])

        # 3. Cross-covariance k_*(x_test, observations)
        print(f"  Building k_star ({Nt}×{Nu+Nf})...")
        k_star = np.zeros((Nt, Nu + Nf))
        for i in range(Nt):
            for j in range(Nu):
                k_star[i, j] = self._k(X_test[i], X_u[j])
            for j in range(Nf):
                k_star[i, Nu + j] = lam * self._d2k_dxp2(X_test[i], X_f[j])

        # 4. Prior variance at test points
        k_star_star = np.array([self._k(X_test[i], X_test[i]) for i in range(Nt)])

        # 5. GP posterior
        print("  Solving GP system...")
        L = np.linalg.cholesky(K_noisy)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        mu = k_star @ alpha

        v = np.linalg.solve(L, k_star.T)
        var = k_star_star - np.sum(v**2, axis=0)

        return mu, var
