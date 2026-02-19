# Workplan: B-PINN Implementation Correction

## 1. Current Status Analysis
The current implementation of the B-PINN is **not functional** and contains critical architectural flaws preventing it from learning or sampling correctly.

### Critical Issues
1.  **Broken Gradient Flow (Critical)**
    *   In `src/models/BNN.py`, the `potential_energy` method calls `self.set_weights(theta)`.
    *   `vector_to_parameters` (used efficiently in `set_weights`) typically operates in-place on leaf tensors (`self.parameters()`), breaking the autograd graph connection to the input `theta`.
    *   **Consequence**: `get_gradient` returns zero or invalid gradients, preventing HMC from exploring the posterior.

2.  **Interface Mismatches**
    *   `src/samplers/HMC.py` calls `model.gradient(theta)` and `model.hamiltonian(theta, r)`.
    *   `src/models/BNN.py` defines `get_gradient` (name mismatch) and lacks `hamiltonian` entirely.
    *   `HMC_sampler` does not accept or pass necessary context arguments (data `x_u, y_u`, physics `pde_problem`) to the model, making evaluation impossible.

3.  **Missing Components**
    *   `src/models/PINN.py` is empty.
    *   `experiments/` folder is empty; there is no entry point to run the code.

## 2. Milestones & Steps

### Milestone 1: Fix Core BNN Architecture
**Goal**: Ensure `BNN` can compute gradients of the potential energy with respect to an input state vector `theta`.

*   **Step 1.1**: Refactor `BNN.forward` to be "functional".
    *   *Approach*: Use `torch.func.functional_call` (PyTorch 2.0+) or manually accept a `weights` dictionary/list in `forward`.
    *   *Why*: This allows evaluating the model using the `theta` tensor from HMC without modifying the model's internal state in-place, preserving the autograd graph.
*   **Step 1.2**: Update `potential_energy` implementation.
    *   Construct the dictionary/params from `theta` (using logic similar to `vector_to_parameters` but differentiable, e.g., using views/reshaping).
    *   Pass these parameters to the functional forward pass.
*   **Step 1.3**: Implement `hamiltonian(theta, r, ...)` in `BNN`.
    *   Formula: $H(\theta, r) = U(\theta) + \frac{1}{2} r^T M^{-1} r$.
    *   Ensure it calls `potential_energy`.

### Milestone 2: Harmonize Sampler and Model
**Goal**: Make `HMC_sampler` and `BNN` compatible.

*   **Step 2.1**: Update `HMC_sampler` signature.
    *   Add `**kwargs` or specific arguments (`data`, `pde_problem`) to `HMC_sampler`.
    *   Pass these arguments to `model.gradient` and `model.hamiltonian` calls.
*   **Step 2.2**: Align Method Names.
    *   Rename `BNN.get_gradient` to `BNN.gradient` to match HMC calls.

### Milestone 3: Implement Functionality & Drivers
**Goal**: Create a runnable experiment.

*   **Step 3.1**: Implement Basic PINN in `src/models/PINN.py`.
    *   Standard MLP with `MSE_u + MSE_f` loss loop (using standard optimizers like Adam/L-BFGS).
*   **Step 3.2**: Create `experiments/run_hmc.py`.
    *   Load Data (e.g., simple 1D example).
    *   Initialize `PDEProblem` (e.g., `Poisson1D`).
    *   Initialize `BNN`.
    *   Run `HMC_sampler`.
    *   Save samples.

## 3. Verification Plan
*   **Unit Test Gradient**: Create a script that acts as a check.
    *   Initialize `BNN` and a random `theta`.
    *   Compute `U = potential_energy(theta, ...)`.
    *   Call `torch.autograd.grad(U, theta)`.
    *   **Success Criteria**: Gradient is non-zero and matches numerical approximation.
*   **Integration Test**: Run HMC for 10 steps on a dummy problem.
    *   Check if `theta` changes.
    *   Check if "acceptance rate" is not always 0 or 1 (indicates reasonable energy transitions).
