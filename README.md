# B-PINNs: Bayesian Physics-Informed Neural Networks

This repository contains the codebase and deliverables for our course project on **Bayesian Physics-Informed Neural Networks (B-PINNs)**, based on the 2020 paper by Yang et al.

The goal of this project is to elucidate the B-PINN methodology, highlight its advantages over standard PINNs in noisy data regimes, and extend the framework through a detailed comparison with Gaussian Processes (GP).

## Team Members
* ** Anouk RUER **
* ** Pénélope FORCIOLI **
* ** Ruben IFRAH **

## Repository Structure

* `data/`: Contains raw and processed datasets (simulated and/or real-world).
* `src/`: Core Python modules.
  * `models/`: Architectures for BNNs and standard PINNs.
  * `physics/`: PDE residual formulations (e.g., Poisson, Diffusion-Reaction).
  * `samplers/`: Implementations of Hamiltonian Monte Carlo (HMC) and Variational Inference (VI).
  * `utils/`: Metrics, visualization, and helper functions.
* `experiments/`: Executable scripts to run specific setups (baseline comparisons, forward problems, inverse problems).
* `notebooks/`: Jupyter notebooks for exploratory data analysis, prototype testing, and generating report figures.
* `deliverables/`: Contains the final NeurIPS-formatted report (LaTeX) and the presentation slides.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[your-username]/bpinn-project.git
   cd bpinn-project
   pip install -r requirements.txt
   ```
   