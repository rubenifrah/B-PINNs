<div align="center">
  <h1>B-PINNs: Bayesian Physics-Informed Neural Networks</h1>
  <p><i>A from-scratch implementation and extension of the B-PINN methodology for robust uncertainty quantification in PDEs.</i></p>
</div>

## Project Context
This project is the **end-of-semester research project** for the **Bayesian Machine Learning** course of the **M2 IASD** program (Université Paris Dauphine - PSL). 

## Project Overview
This repository contains our comprehensive, **from-scratch implementation** of **Bayesian Physics-Informed Neural Networks (B-PINNs)**, initially proposed in the seminal 2020 paper by Yang et al.: *"B-PINNs: Bayesian Physics-Informed Neural Networks for Forward and Inverse PDE Problems with Noisy Data."*

Rather than relying on high-level wrapper libraries, we have built the core Bayesian inference engine—including the **Hamiltonian Monte Carlo (HMC)** sampler and the custom neural network architectures—entirely from the ground up using PyTorch. 

### Goals
Our objective is not just to replicate the findings of Yang et al., but to **push the studies from the paper further**. Specifically, this project aims to:
1. **Elucidate the B-PINN Methodology**: Provide a transparent codebase that demystifies Bayesian inference for PDE solvers.
2. **Robustness in Noisy Regimes**: Rigorously benchmark B-PINNs against standard deterministic PINNs to highlight the Bayesian framework's superior capability in handling highly noisy, sparse sensor data.
3. **Uncertainty Quantification (UQ)**: Demonstrate how B-PINNs provide calibrated confidence intervals ($\pm 2\sigma$) for predictions.
4. **Methodological Extensions (PI-GPs)**: Extend the framework by mathematically linking B-PINNs to Physics-Informed Gaussian Processes (PI-GPs) in the infinite-width limit, and empirically demonstrating why finite networks adapt better to complex dynamics.

---

## Experimental Results & Mini-Report

### 1. The B-PINN Framework
Standard PINNs learn PDE solutions using limited data and physical laws via deterministic gradient descent but lack built-in uncertainty quantification and are highly prone to overfitting when sensor data is noisy. B-PINNs extend this into a Bayesian framework, treating network weights probabilistically to yield a distribution of possible solutions.

<div align="center">
  <img src="report-slides/Figures/intro_pinn_data.png" alt="PINN Data Setup" width="500"/>
  <p><i>Figure 1: Observational data points sampled from the domain and boundaries.</i></p>
</div>

### 2. High Noise Regime (1D Poisson, $\sigma=0.1$)
Under significant sensor noise, standard deterministic PINNs drastically overfit the erratic data points. Our HMC-driven B-PINNs gracefully regularize these predictions, treating the physics loss as a soft constraint that filters high-frequency noise.

| Standard PINN (Overfitting) | Bayesian PINN (Calibrated) | Hierarchical B-PINN |
| :---: | :---: | :---: |
| <img src="report-slides/Figures/1D poisson sigma=0.1/PINN.png" width="300"/> | <img src="report-slides/Figures/1D poisson sigma=0.1/BPINN.png" width="300"/> | <img src="report-slides/Figures/1D poisson sigma=0.1/HBPINN.png" width="300"/> |

*In the high noise regime, the Hierarchical B-PINN (which jointly infers the noise explicitly) acts as an adaptive mathematical "shock absorber", preventing numerical divergence and maintaining robust uncertainty bands.*

### 3. Low Noise Regime (1D Poisson, $\sigma=0.01$)
When noise is low, standard PINNs find sharp, accurate global minima cleanly. B-PINNs maintain excellent coverage (100% PICP) but can occasionally struggle with the highly peaked, non-convex posterior energy landscape typical of sharp data limits. 

| Standard PINN | Bayesian PINN | Hierarchical B-PINN |
| :---: | :---: | :---: |
| <img src="report-slides/Figures/1D poisson sigma = 0.01/PINN.png" width="300"/> | <img src="report-slides/Figures/1D poisson sigma = 0.01/BPINN.png" width="300"/> | <img src="report-slides/Figures/1D poisson sigma = 0.01/HBPINN_modif.png" width="300"/> |

### 4. Theoretical Extension: PI-GPs vs Finite B-PINNs
As part of our extended research, we compared finite-width B-PINNs with infinite-width analytical baselines. By the Central Limit Theorem, BNNs converge to NNGPs, allowing us to derive an exact, sampling-free **Physics-Informed Gaussian Process (PI-GP)** for linear PDEs.

<div align="center">
  <img src="report-slides/Figures/GP/test_gp_result.png" alt="PI-GP Results" width="600"/>
  <p><i>Figure 2: Empirical evaluation of the exact analytical PI-GP on a highly oscillating boundary problem.</i></p>
</div>

**Finding:** While mathematically elegant and possessing an exact closed-form solution (free from HMC sampling instabilities), infinite-width PI-GPs fail severely on complex dynamics. Their rigid fixed kernels cause them to heavily over-smooth the physics in the presence of noise. This proves that **Finite-width B-PINNs are empirically essential**; their active representation learning enables their basis functions to shift and perfectly align with sharp gradients and complex PDE dynamics.

---

## Team Members
* **Anouk RUER**
* **Pénélope FORCIOLI**
* **Ruben IFRAH**

## Repository Structure

* `data/`: Contains raw and processed datasets.
* `src/`: Core Python modules (The from-scratch B-PINN inference engine).
  * `models/`: Architectures for BNNs (`BNN.py`) featuring functional forward passes for autograd-compliant HMC sampling. Includes the extended `BNN_unknown_noise.py` and exact `PIGP.py`.
  * `physics/`: Differentiable PDE residual formulations (e.g., `Poisson1D`, `DampedHarmonicOscillator1D`).
  * `samplers/`: Custom ground-up implementation of Hamiltonian Monte Carlo (`HMC.py`).
  * `utils/`: Rigorous UQ metrics (`PICP`, `MPIW`, `ECE`), visualization tools, and probabilistic plotting helpers.
* `experiments/`: Executable scripts to run specific setups (`run_poisson.py`, `run_damped.py`, etc.).
* `report-slides/`: Contains the final NeurIPS-formatted report (`report.tex`) and the presentation slides (`slides-v2.tex`).

## Setup and Installation

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

To run the full 1D Poisson comparison (Standard PINN vs Bayesian PINN vs Dropouts):
```bash
python experiments/run_poisson.py
```

To run the Damped Harmonic Oscillator baseline:
```bash
python experiments/run_damped.py
```