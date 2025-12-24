import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, 'src'))

from potentials import V_prime_double_well

# ==========================================
# STOCHASTIC CONFIGURATION
# ==========================================
# Langevin parameters: dv = (-gamma*v - V'(u))dt + sigma*dW
SIGMA_NOISE = 0.4  # Nivel de ruido (Temperatura efectiva)
DT = 0.01          # Paso de tiempo fijo para Euler-Maruyama
T_MAX = 20.0       # Tiempo maximo de simulacion

def check_escape_stochastic(u0, v0, gamma):
    """
    Simulates a Langevin trajectory using Euler-Maruyama integration.
    Returns 1 if escape occurs (crosses saddle), 0 otherwise.
    """
    u, v = u0, v0
    steps = int(T_MAX / DT)
    sq_dt = np.sqrt(DT)
    
    # Saddle position + tolerance
    threshold = 0.209 + 0.02
    
    for _ in range(steps):
        # Force and Noise
        force = -V_prime_double_well(u)
        noise = np.random.normal(0, 1)
        
        # Euler-Maruyama Update
        u_next = u + v * DT
        v_next = v + (-gamma * v + force) * DT + SIGMA_NOISE * noise * sq_dt
        
        u, v = u_next, v_next
        
        # Check Escape Condition (Instantaneous crossing)
        if u > threshold:
            return 1 # Escaped
            
    return 0 # Trapped

def compute_basin_stochastic(gamma, res=80):
    """Computes the escape basin grid with noise"""
    u_vals = np.linspace(-1.5, 0.5, res)
    v_vals = np.linspace(-1.0, 2.0, res)
    U, V = np.meshgrid(u_vals, v_vals)
    
    Z = np.zeros_like(U)
    
    # Calculate grid (vectorized where possible, but loop for stochastic)
    for i in range(res):
        for j in range(res):
            Z[i, j] = check_escape_stochastic(U[i, j], V[i, j], gamma)
            
    escape_area = np.sum(Z) / Z.size
    return U, V, Z, escape_area

# ==========================================
# MAIN EXECUTION
# ==========================================
print("="*60)
print(f"[START] STOCHASTIC BASIN MAPPING (Noise sigma={SIGMA_NOISE})")
print("="*60)

gammas = [0.0, 0.16, 0.5]
results = []

for g in gammas:
    print(f"  Calculating gamma={g} with Langevin noise...")
    U, V, Z, area = compute_basin_stochastic(g, res=80)
    results.append((g, Z, area))
    print(f"    -> Escape fraction: {area:.1%}")

print("[OK] ANALYSIS COMPLETE.")

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (g, Z, area) in zip(axes, results):
    # Plot contourf
    # We create a custom colormap or just use RdBu_r
    # Red (1) = Escape, Blue (0) = Trapped
    contour = ax.imshow(Z, origin='lower', extent=[-1.5, 0.5, -1.0, 2.0],
                        cmap='RdBu_r', alpha=0.8, vmin=0, vmax=1)
    
    ax.set_title(f"$\gamma={g}$ (Escape: {area:.1%})")
    ax.set_xlabel(r"Position $\phi$")
    if g == 0.0:
        ax.set_ylabel(r"Velocity $v$")
    
    # Add noise annotation
    ax.text(0.05, 0.95, r"$\xi(t) \neq 0$", transform=ax.transAxes, 
            fontsize=9, color='white', fontweight='bold',
            bbox=dict(facecolor='black', alpha=0.3))

fig.suptitle(f"Stochastic Basin Contraction (Langevin Dynamics, $\sigma={SIGMA_NOISE}$)", fontsize=14)
plt.tight_layout()

# Save
figures_dir = os.path.join(parent_dir, 'figures')
os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, 'fig_basin_collapse.pdf'))
print(f"[OK] Figure saved: {os.path.join(figures_dir, 'fig_basin_collapse.pdf')}")
