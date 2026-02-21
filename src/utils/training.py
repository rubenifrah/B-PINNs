import torch
import torch.optim as optim

from src.samplers.HMC import HMC_sampler

def train_pinn(model, pde_problem, x_b, y_b, x_f, y_f, x_u=None, y_u=None, epochs=1000, lr=1e-3, **kwargs):
    """
    Standard training logic for a baseline Physics-Informed Neural Network (PINN).
    Uses the Adam optimizer to minimize Data and Physics MSE.
    
    Args:
        model: PINN model instance.
        pde_problem: The PDE definition (e.g. Poisson1D).
        x_b, y_b: Boundary conditions (required).
        x_f, y_f: Collocation points and PDE targets.
        x_u, y_u: Optional interior observations.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        
    Returns:
        model: The trained PINN.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting standard PINN optimization for {epochs} epochs...")
    
    history = {"total": [], "mse_b": [], "mse_u": [], "mse_f": []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Compute loss handling both boundary and internal observations natively
        boundary_weight = kwargs.get('boundary_weight', 1.0)
        total_loss, mse_b, mse_u, mse_f = model.compute_loss(x_b, y_b, x_f, pde_problem, x_u=x_u, y_u=y_u, boundary_weight=boundary_weight)
        
        total_loss.backward()
        optimizer.step()
        
        history["total"].append(total_loss.item())
        history["mse_b"].append(mse_b.item())
        history["mse_u"].append(mse_u.item())
        history["mse_f"].append(mse_f.item())
        
        if epoch % max(1, epochs // 10) == 0:
            print(f"Epoch {epoch}: Total Loss {total_loss.item():.5f} | MSE_b {mse_b.item():.5f} | MSE_u {mse_u.item():.5f} | MSE_f {mse_f.item():.5f}")
            
    print("Training complete.")
    return model, history

def train_bpinn(model, pde_problem, x_b, y_b, x_f, y_f, sigma_u=0.1, sigma_f=0.1, x_u=None, y_u=None, theta_0=None, M=100, N=200, L=20, delta_t=0.01, **kwargs):
    """
    Sampling logic for a Bayesian Physics-Informed Neural Network (B-PINN).
    Uses Hamiltonian Monte Carlo (HMC) to map the posterior without standard optimization.
    
    Args:
        model: BNN model instance.
        pde_problem: The PDE definition (e.g. Poisson1D).
        x_b, y_b: Boundary conditions.
        x_f, y_f: Collocation points and PDE targets.
        sigma_u, sigma_f: Standard deviations for likelihood terms.
        x_u, y_u: Optional interior observations.
        theta_0: Initial state (if None, pulled from model weights).
        M: Number of samples to retain.
        N: Total HMC transitions.
        L: Number of leapfrog steps per transition.
        delta_t: Leapfrog step size.
        
    Returns:
        samples: Tensor of shape (num_params, M) holding posterior samples.
    """
    if theta_0 is None:
        theta_0 = model.get_weights()
        
    print(f"Starting B-PINN HMC Sampling with {N} total transitions...")
    
    samples = HMC_sampler(
        model=model,
        M=M,
        N=N,
        delta_t=delta_t,
        theta_0=theta_0,
        L=L,
        x_b=x_b,
        y_b=y_b,
        x_f=x_f,
        y_f=y_f,
        sigma_u=sigma_u,
        sigma_f=sigma_f,
        pde_problem=pde_problem,
        x_u=x_u,
        y_u=y_u
    )
    
    print(f"Sampling complete. Gathered {samples.shape[1]} samples of dimension {samples.shape[0]}.")
    print(f"Diagnostics: Samples Std Dev Mean = {samples.std(dim=1).mean().item():.5f}, Max = {samples.std(dim=1).max().item():.5f}")
    
    return samples
