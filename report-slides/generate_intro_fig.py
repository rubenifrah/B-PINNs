import numpy as np
import matplotlib.pyplot as plt
import os

# Create a figure
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

# Generate a 1D wave (the true solution u)
x = np.linspace(0, 4*np.pi, 200)
z = 0.5 * np.sin(x)

# Plot the underlying wave
ax.plot(x, z, label='True Solution $u(x)$', color='black', alpha=0.5, zorder=1)

# Generate sample observation points (D_u)
# Points inside the domain
u_idx_x = np.random.choice(range(10, 190), 10, replace=False)
ax.scatter(x[u_idx_x], z[u_idx_x] + np.random.normal(0, 0.05, 10), color='red', s=50, label='$\mathcal{D}_u$: Solution Data', zorder=4)

# Generate sample collocation points for forcing term (D_f)
# Plot these as vertical lines or points along the bottom
f_idx_x = np.random.choice(range(5, 195), 30, replace=False)
ax.plot(x[f_idx_x], np.full(30, -0.6), '|', color='blue', markersize=10, label='$\mathcal{D}_f$: Collocation Points', zorder=3)

# Generate boundary points (D_b)
# Points at the edges (x=0, x=4pi)
b_idx_x = [0, 199]
ax.scatter(x[b_idx_x], z[b_idx_x] + np.random.normal(0, 0.02, 2), color='green', marker='s', s=80, label='$\mathcal{D}_b$: Boundary Data', zorder=5)

ax.set_ylim(-0.7, 0.7)
ax.set_xticks([])
ax.set_yticks([])

# Transparent background
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

script_dir = os.path.dirname(os.path.abspath(__file__))
figure_path = os.path.join(script_dir, 'Figures', 'intro_pinn_data.png')

plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.0))
plt.tight_layout()
plt.savefig(figure_path, dpi=300, transparent=True, bbox_inches='tight')

