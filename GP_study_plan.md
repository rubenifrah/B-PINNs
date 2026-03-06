# GP_study_plan.md
## Bridging Bayesian PINNs and Gaussian Processes

### 1. Research Objective
[cite_start]This study extends the original B-PINN framework [cite: 154-155] by formally investigating the infinite-width limit of Bayesian Neural Networks (BNNs). [cite_start]While the original paper relies on finite-width BNNs and Hamiltonian Monte Carlo (HMC) sampling [cite: 158-159], we aim to prove that as the network width approaches infinity, the BNN prior converges to a Neural Network Gaussian Process (NNGP). Consequently, for linear PDEs, the physics-informed constraints map exactly to a Gaussian Process (GP) regression with linear operator observations. We will benchmark the finite B-PINN against this exact analytical PI-GP posterior.

### 2. Phase 1: Theoretical Derivation (The PI-GP Framework)
* **Step 1.1: The NNGP Limit:** Formally define the convergence of the BNN prior to a GP prior as the hidden layer width scales to infinity. 
* **Step 1.2: Linear Operator Stability:** Prove that applying a linear differential operator (e.g., the Laplacian) to a GP yields a joint GP. 
* **Step 1.3: Block-Matrix Formulation:** Derive the exact covariance matrices for the 1D Poisson equation. This requires computing:
    * The data-data covariance.
    * The cross-covariance between the solution and the PDE residual.
    * The physics-physics covariance.
* **Step 1.4: Exact Posterior:** Formulate the closed-form analytical posterior mean and covariance for the Physics-Informed GP (PI-GP).

### 3. Phase 2: Empirical Implementation
* [cite_start]**Step 2.1: Data Generation:** Simulate noisy observations for the 1D Poisson forward problem, creating sets for interior measurements, boundary conditions, and physics collocation points [cite: 583-589].
* **Step 2.2: The Exact GP Regressor:** Implement a custom GP regressor that strictly evaluates the analytical block matrices derived in Step 1.3. This serves as the absolute ground truth.
* **Step 2.3: The Finite B-PINN (HMC):** Implement the B-PINN using the corrected HMC algorithm. [cite_start]**Crucial Fix:** Ensure the Metropolis-Hastings acceptance ratio is properly formulated as the exponential of the old energy minus the new energy, with a strict "less than" inequality for acceptance, correcting the typos present in the original literature's pseudocode [cite: 366-367].
* **Step 2.4: Scaling Experiment:** Run the B-PINN-HMC sampler across strictly increasing hidden layer widths.

### 4. Phase 3: Convergence Analysis & Evaluation
* **Step 3.1: The Derivative Pathology:** Evaluate the prior distributions of the first and second derivatives of the B-PINN. [cite_start]Replicate and expand upon the original findings [cite: 331-333] by showing how the severe non-Gaussianity at narrow widths resolves as the width scales up.
* **Step 3.2: Distance Metrics:** For each network width, compute a statistical distance metric (e.g., Kullback-Leibler divergence or Wasserstein distance) between the empirical HMC posterior and the exact analytical PI-GP posterior.
* **Step 3.3: Visualizing Convergence:** Generate publication-quality plots showing the distance metric decaying as a function of network width.

### 5. Expected Deliverables
* A mathematical proof section in the final NeurIPS-formatted report detailing the PI-GP derivation.
* A custom PI-GP regressor module in the repository.
* Empirical convergence plots bridging the gap between deep learning heuristics and exact statistical theory.