import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.PINN import PINN
from src.models.BNN import BNN
from src.physics.PDEs import PDEProblem
from src.utils.training import train_pinn, train_bpinn

# The true 2D Allen-Cahn problem from the paper:
# u_t - 0.0001 u_xx + 5 u^3 - 5 u = 0
# t in [0, 1], x in [-1, 1]

class AllenCahn2D(PDEProblem):
    def compute_residual(self, u_func, xt, params=None):
        xt.requires_grad_(True)
        u = u_func(xt)
        
        # Gradients
        grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = grads[:, 0:1] # t is the first column
        u_x = grads[:, 1:2] # x is the second column
        
        # Second derivative
        u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 1:2]
        
        return u_t - 0.0001 * u_xx + 5.0 * (u**3) - 5.0 * u - self.y_f

def generate_allen_cahn_data(noise_scale=0.1):
    N_u = 50 
    N_f = 1000 
    
    x_init = np.random.uniform(-1, 1, (N_u, 1))
    t_init = np.zeros((N_u, 1))
    xt_init = np.hstack((t_init, x_init))
    u_init = (x_init**2) * np.cos(np.pi * x_init)
    
    u_init_noisy = u_init + noise_scale * np.random.randn(*u_init.shape)
    
    xt_f = np.random.uniform([0.0, -1.0], [1.0, 1.0], (N_f, 2))
    f_val = np.zeros((N_f, 1))
    f_val_noisy = f_val + noise_scale * np.random.randn(*f_val.shape)
    
    return torch.tensor(xt_init, dtype=torch.float32), torch.tensor(u_init_noisy, dtype=torch.float32), \
           torch.tensor(xt_f, dtype=torch.float32), torch.tensor(f_val_noisy, dtype=torch.float32)

def plot_2d_allen_cahn(model, samples, title="Allen Cahn", save_path="experiments/results/ac_2d.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    t = np.linspace(0, 1, 100)
    x = np.linspace(-1, 1, 100)
    T, X = np.meshgrid(t, x)
    xt_test = torch.tensor(np.hstack((T.flatten()[:,None], X.flatten()[:,None])), dtype=torch.float32)
    
    if samples is not None:
        num_samples = samples.shape[1]
        all_preds = torch.zeros(num_samples, xt_test.shape[0])
        model.eval()
        with torch.no_grad():
            for i in range(num_samples):
                theta = samples[:, i]
                pred = model.functional_forward(theta, xt_test)
                all_preds[i, :] = pred.squeeze()
        u_mean = all_preds.mean(dim=0).numpy().reshape(100, 100)
        u_std = all_preds.std(dim=0).numpy().reshape(100, 100)
    else:
        model.eval()
        with torch.no_grad():
            u_mean = model(xt_test).numpy().reshape(100, 100)
            u_std = np.zeros_like(u_mean)
            
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].pcolormesh(T, X, u_mean, shading='gouraud', cmap='jet')
    axes[0].set_title(f"{title} Mean", fontsize=14)
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    fig.colorbar(im1, ax=axes[0])
    
    if samples is not None:
        im2 = axes[1].pcolormesh(T, X, 2*u_std, shading='gouraud', cmap='jet')
        axes[1].set_title(f"2 Std Dev", fontsize=14)
        axes[1].set_xlabel('t')
        axes[1].set_ylabel('x')
        fig.colorbar(im2, ax=axes[1])
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_allen_cahn_experiment():
    print("Setting up Allen-Cahn 2D...")
    noise_scale = 0.1
    x_b, y_b, x_f, y_f = generate_allen_cahn_data(noise_scale)
    pde_problem = AllenCahn2D(x_f, y_f, noise_scale)
    
    print("Training PINN...")
    pinn = PINN(input_dim=2, output_dim=1, hidden_dims=[50, 50, 50])
    pinn, history = train_pinn(pinn, pde_problem, x_b, y_b, x_f, y_f, epochs=2000, lr=1e-3)
    plot_2d_allen_cahn(pinn, None, "PINN", "experiments/results/ac_pinn.png")
    
    print("Training B-PINN (HMC)...")
    bnn = BNN(input_dim=2, output_dim=1, hidden_dims=[50, 50, 50])
    samples = train_bpinn(bnn, pde_problem, x_b, y_b, x_f, y_f, sigma_u=noise_scale, sigma_f=noise_scale, M=10, N=100, L=10, delta_t=0.0001)
    plot_2d_allen_cahn(bnn, samples, "B-PINN", "experiments/results/ac_bpinn.png")

if __name__ == "__main__":
    run_allen_cahn_experiment()
