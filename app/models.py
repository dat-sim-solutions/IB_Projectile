import numpy as np
from scipy.stats import norm

def run_physics_simulation(m=0.5, A=0.01, g=9.81, rho=1.225, Cd=0.47, dt=0.05, t_max=10,
                           delta_m=0.005, delta_d=0.001, y_uncert_abs=0.40, v_uncert_abs=0.60):
    # 1. Constants
    k = 0.5 * rho * Cd * A
    v_terminal = np.sqrt((m * g) / k)
    t = np.arange(0, t_max, dt)
    
    # 2. Setup Propagation Math
    d = np.sqrt(4 * A / np.pi) 
    uncertainty_A_fract = 2 * (delta_d / d)  
    uncertainty_m_fract = delta_m / m       
    uncertainty_vt_fract = 0.5 * (uncertainty_m_fract + uncertainty_A_fract)  
    v_t_theory_err = v_terminal * uncertainty_vt_fract
    
    # 3. Physics Solutions
    v_ana = v_terminal * np.tanh((g * t) / v_terminal)
    a_ana = g * (1 - np.tanh((g * t) / v_terminal)**2)
    y_ana = (v_terminal**2 / g) * np.log(np.cosh((g * t) / v_terminal))
    
    # Numerical (Euler)
    v_num = np.zeros(len(t)); y_num = np.zeros(len(t)); a_num = np.zeros(len(t))
    for n in range(0, len(t) - 1):
        a_num[n] = g - (k * v_num[n]**2) / m
        v_num[n+1] = v_num[n] + a_num[n] * dt
        y_num[n+1] = y_num[n] + v_num[n] * dt
    a_num[-1] = g - (k * v_num[-1]**2) / m
    
    # 4. Noise Generation (Sigma = Uncertainty / 2)
    a_uncert_abs = ((2 * k * v_ana) / m) * v_uncert_abs
    y_noisy = y_ana + np.random.normal(0, y_uncert_abs / 2, len(t))
    v_noisy = v_ana + np.random.normal(0, v_uncert_abs / 2, len(t))
    a_noisy = a_ana + np.random.normal(0, a_uncert_abs / 2)
    
    residuals = v_noisy - v_ana
    cfl_limit = m / (k * v_terminal)
    
    return {
        "t": t, "v_ana": v_ana, "a_ana": a_ana, "y_ana": y_ana,
        "v_num": v_num, "a_num": a_num, "y_num": y_num,
        "v_noisy": v_noisy, "a_noisy": a_noisy, "y_noisy": y_noisy,
        "v_t_theory_err": v_t_theory_err, "v_uncert_abs": v_uncert_abs,
        "y_uncert_abs": y_uncert_abs, "a_uncert_abs": a_uncert_abs,
        "residuals": residuals, "v_terminal": v_terminal, "cfl_limit": cfl_limit
    }
