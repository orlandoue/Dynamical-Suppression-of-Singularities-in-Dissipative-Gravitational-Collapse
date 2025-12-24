import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import collapse_solver

def run_trace(blueshift):
    # Usamos la misma configuración que funcionó
    sim = collapse_solver.Simulation(gamma=0.01, N=400, pulse_amp=0.15, blue_shift=blueshift)
    vals = []
    times = []
    # Corremos un poco menos de tiempo para evitar la zona inestable
    steps = int(9.5 / sim.dt) 
    
    for i in range(steps):
        sim.step()
        if i % 10 == 0:
            times.append(sim.t)
            # Guardamos la intensidad del campo
            vals.append(np.max(sim.phi**2))
            
    return np.array(times), np.array(vals)

def run_test():
    print("Running Blue-Shift Test (Clean Plot)...")
    t1, v1 = run_trace(False) # Static
    t2, v2 = run_trace(True)  # Blue-shifted
    
    plt.figure(figsize=(8, 5))
    
    # Línea Roja: Sin mecanismo de Tolman (Crece)
    plt.plot(t1, v1, 'r--', linewidth=2, label=r'Static $\gamma = 0.01$ (Insufficient)')
    
    # Línea Azul: Con mecanismo de Tolman (Suprimido)
    plt.plot(t2, v2, 'b-', linewidth=2, label=r'Blue-shifted $\gamma_{eff}(r)$ (Suppressed)')
    
    plt.yscale('log')
    plt.xlim(0, 9.5) # Cortamos antes del error numérico
    plt.ylim(1e-3, 10.0) # Enfocamos en la zona relevante
    
    plt.xlabel(r'Time $t/M$', fontsize=12)
    plt.ylabel(r'Field Intensity $\phi^2$', fontsize=12)
    plt.title(r'Evidence of Tolman Redshift Suppression', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'figures', 'fig_blueshift_test.pdf')
    plt.savefig(out_path)
    print(f"Graph Saved: {out_path}")

if __name__ == "__main__":
    run_test()