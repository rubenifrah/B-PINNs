"""
Diagnostic: Check if each covariance block is actually populated with non-zero values,
and whether the NNGP kernel is differentiable as expected.
"""
import torch
from src.models.PIGP import ExactPIGP
from src.models.NNGP import Erf_NNGP_kernel

sigma_w2 = 1.0
sigma_b2 = 1.0
depth = 2

def my_kernel(x, xp):
    x = x.squeeze()
    xp = xp.squeeze()
    return Erf_NNGP_kernel(x, xp, depth=depth, sigma_w2=sigma_w2, sigma_b2=sigma_b2)

pigp = ExactPIGP(kernel_fn=my_kernel, lambda_pde=0.01, noise_u=0.1, noise_f=0.1)

# --- Test 1: Direct autograd (known working from prior debug) ---
x = torch.tensor([0.3], requires_grad=True)
xp = torch.tensor([0.5], requires_grad=True)
k_val = Erf_NNGP_kernel(x.squeeze(), xp.squeeze(), depth=depth, sigma_w2=sigma_w2, sigma_b2=sigma_b2)
g1 = torch.autograd.grad(k_val, xp, create_graph=True)[0]
g2 = torch.autograd.grad(g1, xp, create_graph=True, allow_unused=True)[0]
print(f"=== Direct autograd ===")
print(f"k(0.3, 0.5) = {k_val.item():.6f}")
print(f"dk/dxp = {g1.item():.6f}")
print(f"d2k/dxp2 = {g2.item() if g2 is not None else 'None'}")

# --- Test 2: Through PIGP methods ---
x2 = torch.tensor([0.3])
xp2 = torch.tensor([0.5])
val_uu = pigp.k_uu(x2, xp2)
val_uf = pigp.k_uf(x2, xp2)
val_fu = pigp.k_fu(x2, xp2)
val_ff = pigp.k_ff(x2, xp2)
print(f"\n=== Through PIGP methods ===")
print(f"k_uu(0.3, 0.5) = {val_uu.item():.6f}")
print(f"k_uf(0.3, 0.5) = {val_uf.item():.6f}  (should be ~0.01 * d2k/dxp2)")
print(f"k_fu(0.3, 0.5) = {val_fu.item():.6f}  (should be ~0.01 * d2k/dx2)")
print(f"k_ff(0.3, 0.5) = {val_ff.item():.6f}  (should be ~0.0001 * d4k/dx2dxp2)")

# --- Test 3: Full block matrix ---
x_b = torch.tensor([[-0.7], [0.7]])
x_f = torch.linspace(-0.7, 0.7, 4).view(-1, 1)  # small for readability
K = pigp.build_covariance_matrix(x_b, x_f)
print(f"\n=== Full block matrix (2 boundary + 4 physics = 6x6) ===")
print(f"K_uu block (2x2):\n{K[:2, :2]}")
print(f"\nK_uf block (2x4):\n{K[:2, 2:]}")
print(f"\nK_fu block (4x2):\n{K[2:, :2]}")
print(f"\nK_ff block (4x4):\n{K[2:, 2:]}")
print(f"\nK_uf max abs value: {K[:2, 2:].abs().max().item():.8f}")
print(f"K_ff max abs value: {K[2:, 2:].abs().max().item():.8f}")
