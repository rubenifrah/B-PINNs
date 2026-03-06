import torch
import math
import matplotlib.pyplot as plt

def compute_relative_l2(y_pred, y_true):
    """
    Computes the Relative L2 Error between the prediction and the ground truth.
    Formula: ||y_pred - y_true||_2 / ||y_true||_2
    """
    # Ensure tensors are flattened to (N,)
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # Calculate the L2 norm of the difference (the error)
    error_norm = torch.linalg.norm(y_pred - y_true, ord=2)
    
    # Calculate the L2 norm of the true solution (the scale)
    true_norm = torch.linalg.norm(y_true, ord=2)
    
    # Return the ratio
    relative_l2 = error_norm / true_norm
    return relative_l2.item()


def evaluate_uncertainty(model, samples, x_test, y_true, n_std=2.0):
    """
    Evaluates PICP and MPIW for the B-PINN posterior samples.
    """
    model.eval()
    predictions = []
    
    # 1. Generate predictions for every sample in the posterior
    for i in range(samples.shape[1]):
        theta = samples[:, i]
        # Use functional_forward to inject the 1D theta vector correctly
        pred = model.functional_forward(theta, x_test).detach()
        predictions.append(pred)
        
    predictions = torch.stack(predictions) # Shape: (M, num_test_points, 1)
    
    # 2. Calculate Mean and Standard Deviation across the M samples
    mu = predictions.mean(dim=0)
    std = predictions.std(dim=0)

    rel_l2 = compute_relative_l2(mu, y_true)

    # 3. Define the uncertainty bounds 
    lower_bound = mu - n_std * std
    upper_bound = mu + n_std * std
    
    # 4. Calculate PICP: Percentage of true points inside the bounds
    in_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
    picp = in_bounds.float().mean().item()
    
    # 5. Calculate MPIW: Average thickness of the uncertainty band
    mpiw = (upper_bound - lower_bound).mean().item()
    

    var = (std ** 2) + 1e-8
    
    # NLL formula: 0.5 * log(2 * pi * sigma^2) + ((y - mu)^2) / (2 * sigma^2)
    nll_per_point = 0.5 * torch.log(2 * math.pi * var) + ((y_true - mu)**2) / (2 * var)
    
    # Take the average across all test points
    mean_nll = nll_per_point.mean().item()

    return picp, mpiw, mean_nll, rel_l2



def compute_ece_and_plot(model, samples, x_test, y_true, num_bins=15, save_path=None):
    """
    Computes the Expected Calibration Error (ECE) for regression 
    and plots a Reliability Diagram.
    """
    model.eval()
    predictions = []
    
    # Generate predictions to get mu and std
    for i in range(samples.shape[1]):
        theta = samples[:, i]
        pred = model.functional_forward(theta, x_test).detach()
        predictions.append(pred)
        
    predictions = torch.stack(predictions)
    mu = predictions.mean(dim=0)
    std = predictions.std(dim=0)
    
    # Define the confidence levels p (e.g., 0.05 to 0.95)
    confidences = torch.linspace(0.05, 0.95, num_bins)
    empirical_coverages = []
    
    # Standard normal distribution to find exact z-scores for each confidence level
    normal_dist = torch.distributions.Normal(0, 1)
    
    for p in confidences:
        # Calculate the z-score for a two-tailed interval of probability p
        z = normal_dist.icdf((1 + p) / 2)
        
        lower_bound = mu - z * std
        upper_bound = mu + z * std
        
        # Calculate actual fraction of points inside this specific interval
        in_bounds = (y_true >= lower_bound) & (y_true <= upper_bound)
        coverage = in_bounds.float().mean().item()
        empirical_coverages.append(coverage)
        
    empirical_coverages = torch.tensor(empirical_coverages)
    
    # ECE is the average absolute difference between predicted confidence and actual coverage
    ece = torch.mean(torch.abs(empirical_coverages - confidences)).item()
    
    # ==========================================
    # Plotting the Reliability Diagram
    # ==========================================
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label="Perfect Calibration")
    plt.plot(confidences.numpy(), empirical_coverages.numpy(), 'bo-', linewidth=2, markersize=6, label="B-PINN Calibration")
    
    # Fill the error gap to visually represent the ECE
    plt.fill_between(confidences.numpy(), confidences.numpy(), empirical_coverages.numpy(), 
                     color='blue', alpha=0.15, label=f"ECE = {ece:.4f}")
    
    plt.xlabel("Expected Confidence Level", fontsize=12)
    plt.ylabel("Observed Empirical Coverage", fontsize=12)
    plt.title("Reliability Diagram (B-PINN)", fontsize=14)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=':', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()
        
    return ece



