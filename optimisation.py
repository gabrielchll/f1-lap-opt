import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import MaximumAcceleration as MaxAcc
import casadi as ca
import TraceTrack as TT

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
f_ = 1.6
g_ = 9.81

class Track :

    def __init__(self, track) :
        db = pd.read_csv(f"tracks/{track}.csv")
        self.name = track
        self.X = np.append(db[["# x_m", "y_m"]].to_numpy(), np.reshape(db[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)                 # Closes the lap
        self.dX = np.diff(self.X, axis = 0)
        self.n = len(self.X)
        self.N = normal(self.X, self.dX, self.n)
        self.ang = np.arctan2(self.dX[:,1], self.dX[:,0])
        self.W = np.append(db[["w_tr_right_m","w_tr_left_m"]].to_numpy(), np.reshape(db[["w_tr_right_m","w_tr_left_m"]].to_numpy()[0], (1,2)), axis = 0)

    def __repr__(self) :
        return self.name

    def initial_speed(self, V0) :                 # Returns initial speed that verify the conditions to help the optimisatiton

        Vx = np.append(V0*np.cos(self.ang), V0*np.cos(self.ang[0]))
        Vy = np.append(V0*np.sin(self.ang), V0*np.sin(self.ang[0]))

        return np.append(Vx.reshape((len(Vx), 1)), Vy.reshape((len(Vy), 1)), axis = 1).flatten()              # Entrelasse les listes

def normal(X, dX, n) : # Returns the normal vector to the middle line in each point of the track, needed to be able to consider the width

    norm = np.sqrt(dX[:,0]**2 + dX[:,1]**2)
    N = np.append(-np.reshape(dX[:,1]/norm,(n-1,1)), np.reshape(dX[:,0]/norm,(n-1,1)), axis = 1)

    return np.append(N, np.reshape(N[0], (1,2)), axis = 0)

def speed(v, n) :   # I give n because v can be np.array or ca.SX so can't determine n directly from it
    
    return ca.sqrt(v[0:2*n:2]**2 + v[1:2*n:2]**2)

def dist_int(track, v) :

    p = position(track, v)
    dp = np.diff(p, axis = 0)

    return ca.sqrt(dp[:,0]**2 + dp[:,1]**2)

def time_int(track, v) :

    V = speed(v, track.n)
    d = dist_int(track, v)

    return 2 * d / (V[1:] + V[0:-1])

def lap_time(track, v) :
    dt = time_int(track, v)
    return ca.sum1(dt)

def heading_change(v, n) :                            # Returns the change of angle of the car at every point of the track.

    cross = v[:2*n-2:2]*v[3:2*n:2] - v[1:2*n-1:2]*v[2:2*n:2]
    dot = v[:2*n-2:2]*v[2:2*n:2] + v[1:2*n-1:2]*v[3:2*n:2]

    return ca.atan2(cross, dot)

def position(track, v) :
    return track.X + ca.vertcat(v[2*track.n::])*track.N

def power_cond(track, v) :
    V = speed(v, track.n)
    dt = time_int(track, v)
    A = (V[1:] - V[0:-1]) / dt
    return A - MaxAcc.a_max(V[:-1])

def noslip_cond(track, v) :

    dt = time_int(track, v)
    V = speed(v, track.n)
    T = heading_change(v, track.n)

    A = (V[1:] - V[:-1])/dt
    theta_der = T/dt 
    
    return (m**2*A**2 + m*A*rho*Cx*S*V[:-1]**2+1/4*(rho*S)**2*(Cx**2-(f_*Cz)**2)*V[:-1]**4 + m**2*V[:-1]**2*theta_der**2 - m*g_*rho*Cz*f_**2*S*V[:-1]**2 - (f_*g_*m)**2)/(f_*g_*m)**2

def pos_cond(track, v) :

    p = position(track, v)
    dp = np.diff(p, axis = 0)

    return v[0:2*track.n-2:2]*dp[:,1] - v[1:2*track.n-2:2]*dp[:,0]          # Cross product needs to be equal to 0 for parallel

def get_problem_nlp(track):

    x = ca.SX.sym('x', 3*track.n)

    return {
        'x': x,
        'f': lap_time(track, x),
        'g': ca.vertcat(power_cond(track, x), noslip_cond(track, x), pos_cond(track, x), speed(x, track.n), x[0] - x[2*(track.n-1)], x[1] - x[2*track.n-1])
    }

def speed_profile(track) :

    Vmax = MaxAcc.v_max()
    
    pb = get_problem_nlp(track)

    solver = ca.nlpsol('nlp', 'ipopt', pb)

    solution = solver(
        x0 = np.append(track.initial_speed(5), np.zeros(track.n)),
        lbx = np.r_[np.full(2*track.n, -np.inf), -track.W[:,0]],
        ubx = np.r_[np.full(2*track.n, np.inf), track.W[:,1]],
        lbg = np.r_[np.full(2*(track.n-1), -np.inf), np.zeros(track.n-1), np.full(track.n, 0.1), 0.0, 0.0],                 # Minimiser la vitesse permet d'éviter la divergence du temps (divisé par v)
        ubg = np.r_[np.zeros(2*(track.n-1)), np.zeros(track.n-1), np.full(track.n, Vmax), 0.0, 0.0])

    print(solver.stats()['return_status']) 
    
    return solution

fig, ax = plt.subplots()

spa = Track("Spa")

sol = speed_profile(spa)
print(sol["f"])

pos = position(spa, sol["x"])

ax.plot(pos[:,0], pos[:,1])
TT.TraceTrack("Spa", ax)

plt.show()