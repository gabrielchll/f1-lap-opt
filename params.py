import numpy as np

g = 9.81             # Gravitational acceleration (m/s^2)
rho = 1.2            # Air density (kg/m^3)

# -------- Estimated parameters for a Formula 1 car following the 2026 regulations --------

R = 0.34             # Wheel radius (m)
i_axle = 4.0         # Final drive ratio
i_gear = np.array([4.25, 3.60, 3.05, 2.60, 2.20, 1.85, 1.57, 1.33])  # Gear ratios
S = 1.4              # Frontal area (m^2)
J = 2*1              # Rotational inertia of the two driven wheels (1 kg*m^2 each)
m = 800              # Car mass (kg)
launch_rpm = 8000    # Launch engine speed (RPM)
Cx = 0.8             # Drag coefficient (constant, well past the drag crisis)
Cz = 3               # Downforce coefficient
mu = 1.6             # Tyre-track friction coefficient
E_max = 4*10**6      # Maximum usable battery energy (J)
P_K_MAX = 350e3      # Maximum MGU-K power (W)
T_K_MAX = 300.0      # Maximum MGU-K torque (N*m)