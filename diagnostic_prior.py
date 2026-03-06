"""
Diagnostic script: Isolate the BNN prior and compare it to the NNGP.
If the HMC sampler can fit sin^3(6x) but the NNGP cannot, then either:
1. The HMC sampler is not actually sampling from the prior (e.g. leapfrog errors, delta_t too large)
2. Initialization `theta_0` is pulling the sampler into a highly probable local mode (breaking detailed balance)
3. The actual empirical distribution of the finite-width BNN is wider than the infinite-width NNGP.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.BNN import BNN
from src.samplers.HMC import HMC_sampler
from src.physics.PDEs import PDEProblem

torch.manual_seed(42)

# Dummy PDE just to pacify the HMC sampler (we will zero out data likelihood)
class DummyPDE(PDEProblem):
    def compute_residual(self, u_func, x, params=None):
        return torch.zeros_like(x)

# Setup 1D points
x = torch.linspace(-0.8, 0.8, 100).view(-1, 1)

# Initialize standard BNN matching the paper
bnn = BNN(input_dim=1, output_dim=1, hidden_dims=[50, 50], prior_std=1.0)

# Sample 10,000 completely random networks (True Empirical Prior)
prior_samples = []
for _ in range(1000):
    # Randomly initialize weights from standard normal (prior_std=1.0)
    for param in bnn.parameters():
        torch.nn.init.normal_(param, mean=0.0, std=1.0)
    with torch.no_grad():
        prior_samples.append(bnn(x).numpy())

prior_samples = np.concatenate(prior_samples, axis=1)
prior_mean = prior_samples.mean(axis=1)
prior_std = prior_samples.std(axis=1)

# Plot the Empirical Prior of the BNN
plt.figure(figsize=(10, 6))
plt.fill_between(x.flatten().numpy(), prior_mean - 2*prior_std, prior_mean + 2*prior_std, color='blue', alpha=0.2, label='±2σ Empirical BNN Prior')
plt.plot(x.flatten().numpy(), prior_mean, 'b-', label='Empirical BNN Mean')

# Overlay the true function we want to fit
u_exact = np.sin(6 * x.flatten().numpy())**3
plt.plot(x.flatten().numpy(), u_exact, 'r--', label='Target: $\sin^3(6x)$')

plt.ylim(-3, 3)
plt.title('Empirical Prior of BNN [50, 50] vs Target Function')
plt.legend()
plt.savefig('diagnostic_prior.png')
print("Saved diagnostic_prior.png")

# Now let's calculate the spectral power (FFT) of the prior samples
# to see if the network is theoretically capable of expressing high frequencies
fft_vals = np.abs(np.fft.rfft(prior_samples, axis=0))
mean_fft = fft_vals.mean(axis=1)
freqs = np.fft.rfftfreq(len(x), d=(1.6/100))

plt.figure(figsize=(10, 6))
plt.plot(freqs, mean_fft, 'b-', label='Mean FFT of BNN Prior Samples')
plt.axvline(x=18/(2*np.pi), color='r', linestyle='--', label='Max Frequency of $\sin^3(6x)$')
plt.yscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude / Spectral Power')
plt.title('Spectral Bias: Frequency Content of BNN Prior')
plt.legend()
plt.savefig('diagnostic_fft.png')
print("Saved diagnostic_fft.png")
