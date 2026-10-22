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
"""m = 800"""
rpm_lance = 8000            # Régime de lancement du moteur (pour le départ)
Cx = 0.8                    # Coefficient environ constant car crise de la trainée déjà atteinte, et la formule 1 n'est jamais en régime laminaire
Cz = 3
f_ = 1.6
g_ = 9.81

def normal(track) : # Returns the normal vector to the middle line in each point of the track, needed to be able to consider the width

    db = pd.read_csv(f"tracks/{track}.csv")
    X = np.append(db[["# x_m", "y_m"]].to_numpy(), np.reshape(db[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)      #closes the lap
    dX = np.diff(X, axis = 0)

    norm = np.sqrt(dX[:,0]**2 + dX[:,1]**2)
    n = len(X)

    N = np.append(-np.reshape(dX[:,1]/norm,(n-1,1)), np.reshape(dX[:,0]/norm,(n-1,1)), axis = 1)

    return np.append(N, np.reshape(N[0], (1,2)), axis = 0)

def angle(dX) :                            # Returns the angle of the car at every point of the track.
    return np.arctan2(dX[:,1], dX[:,0])

def initial_speed(X, V0) :                 # Returns initial speed that verify the conditions to help the optimisatiton

    dX = np.diff(X, axis = 0)
    A = angle(dX)

    Vx = np.append(V0*np.cos(A), V0*np.cos(A[0]))
    Vy = np.append(V0*np.sin(A), V0*np.sin(A[0]))

    return np.append(Vx.reshape((len(Vx), 1)), Vy.reshape((len(Vy), 1)), axis = 1).flatten()              # Entrelasse les listes

def speed(v, n) :
    
    return ca.sqrt(v[0:2*n:2]**2 + v[1:2*n:2]**2)

def dist_int(track, v) :

    p = position(track, v)
    dp = np.diff(p, axis = 0)

    return ca.sqrt(dp[:,0]**2 + dp[:,1]**2)

def time_int(track, v) :

    db = pd.read_csv(f"tracks/{track}.csv")
    X = np.append(db[["# x_m", "y_m"]].to_numpy(), np.reshape(db[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)      #closes the lap
    n = len(X)

    V = speed(v, n)
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

    db = pd.read_csv(f"tracks/{track}.csv")
    X = np.append(db[["# x_m", "y_m"]].to_numpy(), np.reshape(db[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)
    N = normal(track)
    n = len(X)

    return X + ca.vertcat(v[2*n::])*N

def power_cond(track, v, n) :
    V = speed(v, n)
    dt = time_int(track, v)
    A = (V[1:] - V[0:-1]) / dt
    return A - MaxAcc.a_max(V[:-1])

def noslip_cond(track, v, n, m) :

    dt = time_int(track, v)
    V = speed(v, n)
    T = heading_change(v, n)

    A = (V[1:] - V[:-1])/dt
    theta_der = T/dt 
    
    return (m**2*A**2 + m*A*rho*Cx*S*V[:-1]**2+1/4*(rho*S)**2*(Cx**2-(f_*Cz)**2)*V[:-1]**4 + m**2*V[:-1]**2*theta_der**2 - m*g_*rho*Cz*f_**2*S*V[:-1]**2 - (f_*g_*m)**2)/(f_*g_*m)**2

def pos_cond(track, v, n) :

    p = position(track, v)
    dp = np.diff(p, axis = 0)

    return v[0:2*n-2:2]*dp[:,1] - v[1:2*n-2:2]*dp[:,0]

def get_problem_nlp(track, X, n, m):

    Track = pd.read_csv(f"tracks/{track}.csv")
    X = np.append(Track[["# x_m", "y_m"]].to_numpy(), np.reshape(Track[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)
    X = np.append(Track[["# x_m", "y_m"]].to_numpy(), np.reshape(Track[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)      # Completes the lap
    n = len(X)

    x = ca.SX.sym('x', 3*n)

    return {
        'x': x,
        'f': lap_time(track, x),
        'g': ca.vertcat(power_cond(track, x, n), noslip_cond(track, x, n, m), pos_cond(track, x, n), speed(x, n), x[0] - x[2*(n-1)], x[1] - x[2*n-1])
    }

def speed_profile(track, m) :

    Track = pd.read_csv(f"tracks/{track}.csv")
    X = np.append(Track[["# x_m", "y_m"]].to_numpy(), np.reshape(Track[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)      # Completes the lap
    W = np.append(Track[["w_tr_right_m","w_tr_left_m"]].to_numpy(), np.reshape(Track[["w_tr_right_m","w_tr_left_m"]].to_numpy()[0], (1,2)), axis = 0)
    n = len(X)
    Vmax = MaxAcc.v_max()
    
    pb = get_problem_nlp(track, X, n, m)

    solver = ca.nlpsol('nlp', 'ipopt', pb)

    solution = solver(
        x0 = np.append(initial_speed(X, 5), np.zeros(n)),
        lbx = np.r_[np.full(2*n, -np.inf), -W[:,0]],
        ubx = np.r_[np.full(2*n, np.inf), W[:,1]],
        lbg = np.r_[np.full(2*(n-1), -np.inf), np.zeros(n-1), np.full(n, 0.1), 0.0, 0.0],                 # Minimiser la vitesse permet d'éviter la divergence du temps (divisé par v)
        ubg = np.r_[np.zeros(2*(n-1)), np.zeros(n-1), np.full(n, Vmax), 0.0, 0.0])

    print(solver.stats()['return_status']) 
    
    return solution

def dist(track) :

    Raceline = pd.read_csv(f"tracks/{track}.csv").to_numpy()
    X = Raceline[:,0]
    Y = Raceline[:,1]
    N = len(X)

    d = [0]

    for i in range(0, N-1) :
        norm = np.sqrt((X[i+1] - X[i])**2 + (Y[i+1] - Y[i])**2)
        d.append(d[i] + norm)
    norm = np.sqrt((X[0] - X[N-1])**2 + (Y[0] - Y[N-1])**2)
    d.append(d[N-1] + norm)

    return d

"""fig, ax = plt.subplots()

fig, ax1 = plt.subplots()

TT.TraceTrack("Suzuka", ax)

V = speed_profile("Suzuka")

v = V["x"].full().flatten()
D = dist("Suzuka")
X = position("Suzuka", v)

Speed = speed(v, len(D))

ax1.plot(D, Speed*3.6)
print(V["f"])

TT.TraceTrack("Suzuka", ax)
ax.plot(X[:,0], X[:,1], color = "red", lw = 0.4)"""

M = [600 + 10*i for i in range(0, 30)]
t = []

for m in M :    
    t.append(float(speed_profile("Suzuka", m)["f"]))

plt.plot(M, t)

plt.show()
