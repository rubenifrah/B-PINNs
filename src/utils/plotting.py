import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_1d_pinn(model, x_u, y_u, x_f, y_f=None, true_solution_func=None, title="1D PINN Prediction", save_path="experiments/results/pinn_1d_result.png"):
    """
    Plots the predictions of a trained PINN for any 1D problem.
    Optionally compares the neural network prediction with a true analytical solution
    and visualizes the PDE forcing targets (y_f) on a secondary axis.
    
    Args:
        model: Trained PINN model.
        x_u: Boundary/Initial data points (tensor).
        y_u: Boundary/Initial data values (tensor).
        x_f: Collocation points (tensor).
        y_f: Optional. PDE Target values at collocation points (tensor).
        true_solution_func: Optional callable that takes a numpy array `x` and returns `u_true`.
        title: Title for the plot.
        save_path: Full path to save the generated plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate continuous test points over the domain
    x_min = float(x_f.detach().cpu().min()) 
    x_max = float(x_f.detach().cpu().max()) 
    
    margin = (x_max - x_min) * 0.05
    x_test = torch.linspace(x_min - margin, x_max + margin, 200).view(-1, 1)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        u_pred = model(x_test).numpy()
    
    # Convert tensors to numpy for plotting
    x_test_np = x_test.numpy()
    x_u_np = x_u.numpy()
    y_u_np = y_u.numpy()
    x_f_np = x_f.detach().numpy()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot true solution on primary axis if provided
    if true_solution_func is not None:
        u_true = true_solution_func(x_test_np)
        ax1.plot(x_test_np, u_true, 'k-', label='True Solution $u(x)$', linewidth=2)
    
    # Plot PINN prediction on primary axis
    ax1.plot(x_test_np, u_pred, 'r--', label='Predictive mean $\hat{u}(x)$', linewidth=2)
    
    # Plot boundary/observation data on primary axis
    ax1.scatter(x_u_np, y_u_np, color='blue', s=100, label='Noisy observations', zorder=5)
    
    # If no y_f provided, optional collocation points plotting
    # if y_f is None: ...
    
    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x)', fontsize=14)
    # ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=12)
    
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    
    # Explicitly close to free memory
    plt.close()

def plot_1d_bpinn(model, samples, x_u, y_u, x_f, y_f=None, true_solution_func=None, title="1D B-PINN Prediction (Uncertainty)", save_path="experiments/results/bpinn_1d_result.png"):
    """
    Plots the predictions of a trained Bayesian PINN with uncertainty bounds.
    
    Args:
        model: BNN model with `functional_forward(theta, x)` method.
        samples: Tensor of shape (num_params, M) containing HMC samples.
        x_u: Boundary/Initial data points (tensor).
        y_u: Boundary/Initial data values (tensor).
        x_f: Collocation points (tensor).
        y_f: Optional. PDE Target values at collocation points (tensor).
        true_solution_func: Optional callable for the true solution.
        title: Title for the plot.
        save_path: Full path to save the generated plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    x_min = float(torch.min(x_f))
    x_max = float(torch.max(x_f))
    margin = (x_max - x_min) * 0.05
    x_test = torch.linspace(x_min - margin, x_max + margin, 200).view(-1, 1)
    
    # Generate predictions for all samples
    num_samples = samples.shape[1]
    all_preds = torch.zeros(num_samples, x_test.shape[0])
    
    model.eval()
    with torch.no_grad():
        for i in range(num_samples):
            theta = samples[:, i]
            pred = model.functional_forward(theta, x_test)
            all_preds[i, :] = pred.squeeze()
            
    # Calculate mean and standard deviation
    u_mean = all_preds.mean(dim=0).numpy()
    u_std = all_preds.std(dim=0).numpy()
    
    x_test_np = x_test.numpy().flatten()
    x_u_np = x_u.numpy().flatten()
    y_u_np = y_u.numpy().flatten()
    x_f_np = x_f.detach().numpy().flatten()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot true solution
    if true_solution_func is not None:
        u_true = true_solution_func(x_test_np)
        ax1.plot(x_test_np, u_true, 'k-', label='True Solution $u(x)$', linewidth=2)
        
    # Plot predictive mean
    ax1.plot(x_test_np, u_mean, 'r--', label='Posterior mean', linewidth=2)
    
    # Plot uncertainty bounds
    ax1.fill_between(x_test_np, u_mean - 2*u_std, u_mean + 2*u_std,
                    alpha=0.3, color='cyan', label='±2 std')
    
    # Plot boundary/observation data
    ax1.scatter(x_u_np, y_u_np, color='blue', s=100, label='Noisy observations', zorder=5)
    
    # Optional PDE Targets (noisy y_f) - currently not standard for this style, but keeping if requested. 
    # Usually we don't plot this for the simple B-PINN plot from train_bpinn_unknown_noise.py
    # if y_f is not None:
    #     ...

    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('u(x)', fontsize=14)
    # ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, linestyle='-', alpha=0.3)
    ax1.legend(loc='upper right', fontsize=12)
        
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    plt.close()

def plot_loss_curves(history, save_path="experiments/results/loss_curves.png"):
    """
    Plots the training loss curves for standard PINNs.
    
    Args:
        history: Dictionary containing loss arrays ('total', 'mse_b', 'mse_u', 'mse_f')
        save_path: Full path to save the generated plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(len(history['total']))
    
    plt.figure(figsize=(10, 6))
    
    # Use log scale for loss
    plt.semilogy(epochs, history['total'], 'k-', label='Total Loss', linewidth=2)
    plt.semilogy(epochs, history['mse_f'], 'g--', label='Physics Loss (MSE_f)', alpha=0.8)
    
    if 'mse_b' in history and any(v > 0 for v in history['mse_b']):
        plt.semilogy(epochs, history['mse_b'], 'r:', label='Boundary Loss (MSE_b)', alpha=0.8)
        
    if 'mse_u' in history and any(v > 0 for v in history['mse_u']):
        plt.semilogy(epochs, history['mse_u'], 'b-.', label='Observation Loss (MSE_u)', alpha=0.8)
        
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss (Log Scale)', fontsize=14)
    plt.title('PINN Training Loss Curves', fontsize=16)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Loss curves plot saved successfully to: {save_path}")
    plt.close()
