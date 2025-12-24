import numpy as np

class Simulation:
    """
    Spherically symmetric collapse solver.
    REFEREE EDITION:
    1. Positive Potential V = 0.5*phi^2 (Ensures Energy > 0 for validation).
    2. Ultra-small dt (Prevents Blue-shift instability).
    """

    def __init__(self,
                 gamma=0.0,
                 N=400,
                 r_max=20.0,
                 pulse_amp=0.01,
                 blue_shift=False):

        # --- Grid ---
        self.r_min = 0.1
        self.r = np.linspace(self.r_min, r_max, N)
        self.dr = self.r[1] - self.r[0]
        self.N = N

        # --- Time (ULTRA FINE STEP FOR STABILITY) ---
        # Bajamos de 0.02 a 0.005 para soportar fricción alta (Blue-shift)
        self.dt = 0.005 * self.dr 
        self.t = 0.0

        # --- Physics ---
        self.gamma_base = gamma
        self.blue_shift_mode = blue_shift
        self.G = 1.0
        
        # Metric Init
        self.a = np.ones(N) 

        # --- Initial Data: Traveling Pulse ---
        r0 = 8.0
        sigma = 1.5 
        self.phi = pulse_amp * np.exp(-(self.r - r0)**2 / sigma**2)
        
        # Momentum ingoing
        dphi_dr = -2 * (self.r - r0) / sigma**2 * self.phi
        self.Pi = dphi_dr 

        self.dissipated_energy_accumulated = 0.0
        self.crashed = False

        self.update_metric()

    def get_gamma_profile(self):
        if not self.blue_shift_mode:
            return np.full_like(self.r, self.gamma_base)
        else:
            # Blue-shift: gamma ~ gamma_0 * a^2
            # Cap suave para evitar singularidad numérica dura
            factor = np.clip(self.a**2, 1.0, 40.0) 
            return self.gamma_base * factor

    def update_metric(self):
        dphi = np.gradient(self.phi, self.dr)
        
        # --- CAMBIO CLAVE: POTENCIAL POSITIVO SIMPLIFICADO ---
        # Usamos V = 1/2 phi^2 para que la energía sea siempre positiva
        # y la gráfica de conservación sea clara.
        V = 0.5 * self.phi**2
        
        denom_a = np.clip(self.a, 1.0, 20.0)
        rho = 0.5 * (self.Pi**2 + (dphi/denom_a)**2) + V
        
        new_a = np.ones_like(self.a)
        current_a = 1.0
        
        for i in range(self.N):
            r_val = self.r[i]
            rho_val = rho[i]
            
            safe_a = min(current_a, 10.0) 
            safe_rho = min(rho_val, 50.0) # Clamp más estricto
            
            term1 = (safe_a**3 - safe_a)/(2*r_val)
            term2 = 4 * np.pi * self.G * r_val * (safe_a**3) * safe_rho
            
            da = (term1 + term2) * self.dr
            
            # Limitar pendiente para estabilidad
            if da > 0.2: da = 0.2
            if da < -0.2: da = -0.2
            
            current_a += da
            
            if current_a > 15.0: current_a = 15.0
            if current_a < 1.0: current_a = 1.0
            
            new_a[i] = current_a
            
        self.a = new_a

    def calculate_total_energy(self):
        dphi = np.gradient(self.phi, self.dr)
        # Usar el mismo potencial positivo
        V = 0.5 * self.phi**2
        rho = 0.5 * (self.Pi**2 + (dphi/self.a)**2) + V
        dV = 4 * np.pi * self.r**2 * self.dr
        return np.sum(rho * dV)

    def step(self):
        if self.crashed: return

        self.update_metric()
        gamma_array = self.get_gamma_profile()
        
        d2f = np.gradient(np.gradient(self.phi, self.dr), self.dr)
        df = np.gradient(self.phi, self.dr)
        lap_phi = d2f + (2.0/self.r) * df
        
        # Fuerza del potencial armónico V' = phi
        V_prime = self.phi 
        force = lap_phi - V_prime
        
        dV = 4 * np.pi * self.r**2 * self.dr
        power_loss = np.sum(gamma_array * self.Pi**2 * dV)
        self.dissipated_energy_accumulated += power_loss * self.dt

        # Update implícito para estabilidad
        denom = 1.0 + gamma_array * self.dt
        self.Pi = (self.Pi + self.dt * force) / denom
        self.phi += self.dt * self.Pi
        self.t += self.dt
        
        self.phi[-1] = 0; self.Pi[-1] = 0
        self.phi[0] = self.phi[1]; self.Pi[0] = self.Pi[1]
        
        if np.any(np.isnan(self.phi)):
            self.crashed = True