<div align="center">
  <h1>B-PINNs: Bayesian Physics-Informed Neural Networks</h1>
  <p><i>A from-scratch implementation and extension of the B-PINN methodology for robust uncertainty quantification in PDEs.</i></p>
</div>

##  Project Overview

This repository contains our comprehensive, **from-scratch implementation** of **Bayesian Physics-Informed Neural Networks (B-PINNs)**, initially proposed in the seminal 2020 paper by Yang et al.: *"B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and Inverse PDE Problems with Noisy Data."*

Rather than relying on high-level wrapper libraries, we have built the core Bayesian inference engine—including the Hamiltonian Monte Carlo (HMC) sampler and the custom neural network architectures—entirely from the ground up using PyTorch. 

###  Goals

Our objective is not just to replicate the findings of Yang et al., but to **push the studies from the paper further**. Specifically, this project aims to:
1. **Elucidate the B-PINN Methodology:** Provide a transparent, easy-to-follow codebase that demystifies Bayesian inference in the context of PDE solvers.
2. **Robustness in Noisy Regimes:** Rigorously benchmark B-PINNs against standard PINNs to highlight the Bayesian framework's superior capability in handling highly noisy, sparse sensor data.
3. **Uncertainty Quantification (UQ):** Demonstrate how B-PINNs provide calibrated confidence intervals for predictions, a critical feature missing in deterministic PINNs.
4. **Methodological Extensions:** Extend the original framework through detailed empirical comparisons with alternative probabilistic surrogate models, such as Gaussian Processes (GPs).

## 👥 Team Members
* **Anouk RUER**
* **Pénélope FORCIOLI**
* **Ruben IFRAH**

##  Repository Structure

* `data/`: Contains raw and processed datasets (simulated and/or real-world).
* `src/`: Core Python modules (The from-scratch B-PINN engine).
  * `models/`: Architectures for BNNs (`BNN.py`) featuring functional forward passes for autograd-compliant HMC sampling, and standard `PINN.py` baselines.
  * `physics/`: Differentiable PDE residual formulations (e.g., `Poisson1D`).
  * `samplers/`: Custom implementations of **Hamiltonian Monte Carlo (HMC)**.
  * `utils/`: Metrics, visualization tools, and probabilistic plotting helpers.
* `experiments/`: Executable scripts to run specific setups (e.g., `train_binn.py`, `train_pinn.py`).
* `notebooks/`: Jupyter notebooks for exploratory data analysis, prototype testing, and generating report figures.
* `deliverables/`: Contains the final NeurIPS-formatted report (LaTeX) and the presentation slides.

## 🚀 Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rubenifrah/B-PINNs.git
   cd B-PINNs
   ```

2. **Install dependencies:**
   Ensure you have Python 3.8+ installed.
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

To run the baseline standard PINN optimization:
```bash
python experiments/train_pinn.py
```

To run the from-scratch Bayesian PINN using our custom HMC sampler:
```bash
python experiments/train_binn.py
```