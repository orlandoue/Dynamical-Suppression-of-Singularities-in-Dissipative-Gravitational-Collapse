import numpy as np
from scipy.integrate import solve_ivp
from scipy.ndimage import gaussian_filter

def invariant_measure(gamma, V_prime_func, n_traj=50, T_total=500.0, 
                      burnin=100.0, domain=(-2, 2), bins=100):
    """Calcula medida invariante por muestreo de trayectorias"""
    phi_min, phi_max = domain
    v_min, v_max = domain
    
    hist = np.zeros((bins, bins))
    phi_edges = np.linspace(phi_min, phi_max, bins + 1)
    v_edges = np.linspace(v_min, v_max, bins + 1)
    
    for _ in range(n_traj):
        # Condición inicial aleatoria
        phi0 = np.random.uniform(phi_min, phi_max)
        v0 = np.random.uniform(v_min, v_max)
        
        # Integrar larga trayectoria
        t_eval = np.linspace(burnin, T_total, 1000)
        sol = solve_ivp(
            lambda t, y: [y[1], -gamma*y[1] - V_prime_func(y[0])],
            [0, T_total],
            [phi0, v0],
            t_eval=t_eval,
            method='Radau',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Histogramar
        for phi, v in zip(sol.y[0], sol.y[1]):
            i = np.digitize(phi, phi_edges) - 1
            j = np.digitize(v, v_edges) - 1
            if 0 <= i < bins and 0 <= j < bins:
                hist[i, j] += 1
    
    return hist / hist.sum()

def effective_support(H, eps=1e-5):
    """Fracción del espacio con medida > eps"""
    return np.sum(H > eps) / H.size

def fisher_information(H, eps=1e-12):
    """Información de Fisher para distribución 2D"""
    H = H + eps
    H = H / H.sum()
    logH = np.log(H)
    
    # Derivadas centradas
    dlog_dx = (np.roll(logH, -1, axis=0) - np.roll(logH, 1, axis=0)) / 2.0
    dlog_dy = (np.roll(logH, -1, axis=1) - np.roll(logH, 1, axis=1)) / 2.0
    
    return np.sum((dlog_dx**2 + dlog_dy**2) * H)

def perturb_measure(H, noise=0.05, sigma=1.0):
    """Perturba medida para test variacional"""
    perturbed = gaussian_filter(H, sigma=sigma)
    perturbed += noise * np.random.randn(*H.shape)
    perturbed = np.maximum(perturbed, 0)
    return perturbed / perturbed.sum()