import numpy as np
import params as p

# ---------- Hybrid Formula 1 powertrain model inspired by the 2026 regulations  ----------

_rpm_pts = np.array([4000, 6000, 8000, 10500, 12000, 13000, 15000])
_T_ICE_pts   = np.array([ 125,  190,  240,   250,   220,   205,   170])  # N·m

V_ = np.linspace(1, 95, 95)
V_t = V_[25:]    # Speeds on which we fit a polynome, we do not take into account lower speeds because polynomial does not fit well and acceleration is limited by the no slip condition anyways

def ICE_torque_map(rpm):
    rpm = np.asarray(rpm, dtype=float)
    return np.where(rpm > 15000, 0.0, np.interp(rpm, _rpm_pts, _T_ICE_pts))

def ICE_power_map(rpm):
    return ICE_torque_map(rpm) * (np.asarray(rpm) * 2*np.pi/60) / 1e3

def OptGear(V) :        # Returns the optimal gear of the car at a given speed V by maximising the available power
          
    rpm = ((V*60)/(p.R*2*np.pi)) * p.i_gear * p.i_axle
    P_max = 0
    i_max = 0

    for i in range(0, len(p.i_gear)) :
        P = ICE_power_map(rpm[i])
        if P > P_max :
            P_max = P
            i_max = i

    return i_max

def engine_speed(V) :
    n = OptGear(V)
    rpm_mot = ((V*60)/(p.R*2*np.pi)) * p.i_gear[n] * p.i_axle
    if n == 0 :
        return max(rpm_mot, p.launch_rpm)      # Accounts for clutch slip (not currently used in the model but will come in handy later on)
    return rpm_mot

def MGUK_torque_map(rpm):
    rpm = np.asarray(rpm, dtype=float)
    w = np.maximum(rpm, 1.0) * 2*np.pi/60
    return np.minimum(p.T_K_MAX, p.P_K_MAX / w)

def max_torque_ICE(V) :
    return ICE_torque_map(engine_speed(V))*p.i_gear[OptGear(V)]*p.i_axle

def max_torque_MGUK(V) :
    return MGUK_torque_map(engine_speed(V)) * p.i_axle * p.i_gear[OptGear(V)]

def polynome(c, x) :
    pol = 0
    for i in range(0, len(c)) :
        pol = pol + c[i]*x**(len(c) - 1 - i)
    return pol

# Computes 5th-degree polynomial fits
C_MGUK = np.polyfit(V_t, [max_torque_MGUK(V_t[i]) for i in range(0, len(V_t))], 5)
C_ICE = np.polyfit(V_t, [max_torque_ICE(V_t[i]) for i in range(0, len(V_t))], 5)

def T_MGUK_max(x) :
    return polynome(C_MGUK, x)

def T_ICE_max(x) :
    return polynome(C_ICE, x)
