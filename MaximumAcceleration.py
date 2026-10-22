import numpy as np
import matplotlib.pyplot as plt
import casadi as ca
import random as rd

L=5.5
nu=1.5e-5
R = 0.34          
i_pont = 4.0           
i_boite = np.array([4.25, 3.60, 3.05, 2.60, 2.20, 1.85, 1.57, 1.33])
rho = 1.2
S   = 1.4
J = 2*1                     # Le moteur entraine deux roues d'inerties 1 kg/m**2
m = 800
rpm_lance = 8000            # Régime de lancement du moteur (pour le départ)
Cx = 0.8                    # Coefficient environ constant car crise de la trainée déjà atteinte, et la formule 1 n'est jamais en régime laminaire
Cz = 3
f = 1.6
g = 9.81

# ---------- Couple moteur (V6 turbo 1.6 L, thermique seul) ----------

_rpm_pts = np.array([4000, 6000, 8000, 10500, 12000, 13000, 15000])
_C_pts   = np.array([ 250,  380,  480,   500,   440,   410,   340])  # N·m

def couple(rpm):
    rpm = np.asarray(rpm, dtype=float)
    return np.where(rpm > 15000, 0.0, np.interp(rpm, _rpm_pts, _C_pts))

def puissance(rpm):
    return couple(rpm) * (np.asarray(rpm) * 2*np.pi/60) / 1e3

V = np.array([1 + i for i in range(0,95)])

def OptGear(V) :                # Returns the optimal gear given the speed
          
    rpm = ((V*60)/(R*2*np.pi)) * i_boite * i_pont
    P_max = 0
    i_max = 0

    for i in range(0, 8) :
        P = puissance(rpm[i])
        if P > P_max :
            P_max = P
            i_max = i

    return i_max

def regime_moteur(V) :
    n = OptGear(V)
    rpm_mot = ((V*60)/(R*2*np.pi)) * i_boite[n] * i_pont
    if n == 0 :
        return max(rpm_mot, rpm_lance)      # Takes in account the clutch
    return rpm_mot
        
def MaxAcceleration(V) :
    return (1/((J/R)+R*m)) * ((i_pont * i_boite[OptGear(V)]) * couple(regime_moteur(V)) - ((R)/2)*rho*Cx*S*V**2)

def fin_embrayage() :
    return R*rpm_lance*2*np.pi/(i_boite[0]*60)

# Les 30 m.s-2 ne seront jamais atteint et son limité par le glissement donc on veut une régression sur A seulement pour des vitesses > 25 m.s-2
# Avec une fonction polynomiale et une loss quadratique, ça revient à calculer le vecteur en lequel la distance de Y à l'espace vectoriel image de P est atteind

def polynome(c, x) :
    pol = 0
    for i in range(0, len(c)) :
        pol = pol + c[i]*x**(len(c) - 1 - i)
    return pol

def a_max(v) :

    V_t = np.array([25 + i for i in range(0, 80)])
    A_t = np.array([MaxAcceleration(V_t[i]) for i in range(0, len(V_t))])

    coeffs = np.polyfit(V_t, A_t, 5)

    return polynome(coeffs, v)

V_t = np.array([25 + i for i in range(0, 80)])
A_t = np.array([MaxAcceleration(V_t[i]) for i in range(0, len(V_t))])

def v_max(coeffs = np.polyfit(V_t, A_t, 5), v_lo=25.0, v_hi=125):
    r = np.roots(coeffs)
    r = r[np.abs(r.imag) < 1e-9].real
    r = r[(r >= v_lo) & (r <= v_hi)]
    return np.min(r)