import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import powertrain as pwt
import casadi as ca
import params as p

class Track:
    """Contains all the track's caracteristics."""

    def __init__(self, track):
        db = pd.read_csv(f"tracks/{track}.csv")
        self.name = track
        self.X = np.append(db[["# x_m", "y_m"]].to_numpy(), np.reshape(db[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)                 # Closes the lap
        self.dX = np.diff(self.X, axis = 0)
        self.n = len(self.X)
        self.N = normal(self.dX, self.n)
        self.ang = np.arctan2(self.dX[:,1], self.dX[:,0])
        self.W = np.append(db[["w_tr_right_m","w_tr_left_m"]].to_numpy(), np.reshape(db[["w_tr_right_m","w_tr_left_m"]].to_numpy()[0], (1,2)), axis = 0)

    def __repr__(self):
        return self.name

    def initial_speed(self, V0):
        """Returns an initial speed profile that verifies the track constraints for a proper initialisation of the optimisation"""

        Vx = np.append(V0*np.cos(self.ang), V0*np.cos(self.ang[0]))
        Vy = np.append(V0*np.sin(self.ang), V0*np.sin(self.ang[0]))

        return np.append(Vx.reshape((len(Vx), 1)), Vy.reshape((len(Vy), 1)), axis = 1).flatten()        # Intertwines the x and y speeds

    def trace_track(self, ax):
        """Traces the tracks outline"""
        
        X_l = self.X + np.reshape(self.W[:,1], (self.n, 1)) * self.N
        X_r = self.X - np.reshape(self.W[:,0], (self.n, 1)) * self.N
        ax.plot(X_l[:,0], X_l[:,1], color = "black", lw = 0.4)
        ax.plot(X_r[:,0], X_r[:,1], color = "black", lw = 0.4)
        ax.axis("equal")


def normal(dX, n):
    """Returns the normal vector to the middle line in each point of the track using the dot product"""

    norm = np.sqrt(dX[:,0]**2 + dX[:,1]**2)
    N = np.append(-np.reshape(dX[:,1]/norm,(n-1,1)), np.reshape(dX[:,0]/norm,(n-1,1)), axis = 1)

    return np.append(N, np.reshape(N[0], (1,2)), axis = 0)


def unpack(x, n):
    """Split the decision vector x (size 6n) into named blocks of n nodes."""
    return {
        "vx": x[0:2*n:2],
        "vy": x[1:2*n:2],
        "d": x[2*n:3*n],
        "T_ice": x[3*n:4*n],
        "T_mguk": x[4*n:5*n],
        "E": x[5*n:6*n],
    }


def speed(v, n):
    x = unpack(v, n)
    return ca.sqrt(x["vx"]**2 + x["vy"]**2)


def energy_constraint(track, v):
    """Relationship between MGU-K torque and the energy level of the battery. Constrained to = 0."""

    x = unpack(v, track.n)
    dx = dist_int(track, v)
    energy_var = -x["T_mguk"][:-1]*dx/p.R

    return (x["E"][1:] - x["E"][:-1]) - energy_var


def dist_int(track, v):
    """Returns the distance interval between two successive positions of the car"""

    pos = position(track, v)
    dp = pos[1:, :] - pos[:-1, :]

    return ca.sqrt(dp[:,0]**2 + dp[:,1]**2)


def time_int(track, v):
    """Returns the non-constant time interval"""

    V = speed(v, track.n)
    d = dist_int(track, v)

    return 2 * d / (V[1:] + V[0:-1])


def lap_time(track, v):
    dt = time_int(track, v)
    return ca.sum1(dt)


def heading_change(v, n):
    """Returns the change of angle of the car's heading direction between two successive positions of the car"""

    x = unpack(v, n)

    cross = x["vx"][:-1] * x["vy"][1:] - x["vy"][:-1] * x["vx"][1:]
    dot = x["vx"][:-1] * x["vx"][1:] + x["vy"][:-1] * x["vy"][1:]

    return ca.atan2(cross, dot)


def position(track, v):
    """Returns the position of the car on track in the tracks reference frame"""

    x = unpack(v, track.n)
    
    return track.X + ca.vertcat(x["d"])*track.N


def power_cond(track, v):
    """Dynamic equation linking the ICE's, MGU-K's and brake's torques and the speed of the car. Constrained to <= 0."""

    x = unpack(v, track.n)
    V = speed(v, track.n)
    dt = time_int(track, v)
    A = (V[1:] - V[0:-1]) / dt

    return ((p.J/p.R + p.R*p.m)*A - x["T_ice"][:-1] - x["T_mguk"][:-1] + 1/2*p.R*p.rho*p.S*p.Cx*V[:-1]**2)/1000


def noslip_cond(track, v):
    """Friction-circle constraint, normalized by (mu*g*m)^2. Constrained to <= 0."""

    dt = time_int(track, v)
    V = speed(v, track.n)
    T = heading_change(v, track.n)
    A = (V[1:] - V[:-1])/dt
    theta_der = T/dt 
    
    return (p.m**2*A**2 + p.m*A*p.rho*p.Cx*p.S*V[:-1]**2+1/4*(p.rho*p.S)**2*(p.Cx**2-(p.mu*p.Cz)**2)*V[:-1]**4 + p.m**2*V[:-1]**2*theta_der**2 - p.m*p.g*p.rho*p.Cz*p.mu**2*p.S*V[:-1]**2 - (p.mu*p.g*p.m)**2)/(p.mu*p.g*p.m)**2


def pos_cond(track, v):
    """Cross product between position variation vector and speed vector. Constrained to = 0."""

    x = unpack(v, track.n)
    pos = position(track, v)
    dp = pos[1:, :] - pos[:-1, :]

    return x["vx"][:-1]*dp[:,1] - x["vy"][:-1]*dp[:,0]


def solve_lap(track):
    """Solve the minimum-lap-time NLP with Ipopt and return the CasADi solution.

    Decision vector x (size 6n), stored in blocks of n nodes:
    [vx, vy interleaved (2n) | lateral offset d | T_ICE | T_MGU-K | battery E]
    """

    x = ca.SX.sym('x', 6*track.n)
    
    pb = {
        'x': x,
        'f': lap_time(track, x),
        'g': ca.vertcat(power_cond(track, x), noslip_cond(track, x), pos_cond(track, x), x[3*track.n:4*track.n] - pwt.T_ICE_max(speed(x, track.n)), x[4*track.n:5*track.n] - pwt.T_MGUK_max(speed(x, track.n)), x[4*track.n:5*track.n] + pwt.T_MGUK_max(speed(x, track.n)), energy_constraint(track, x), x[0] - x[2*(track.n-1)], x[1] - x[2*track.n-1])
    }

    solver = ca.nlpsol('nlp', 'ipopt', pb)

    solution = solver(
        x0 = np.r_[track.initial_speed(5), np.zeros(3*track.n), np.full(track.n, p.E_max)],
        lbx = np.r_[np.full(2*track.n, -np.inf), -track.W[:,0], np.zeros(track.n), np.full(track.n, -np.inf), np.zeros(track.n)],
        ubx = np.r_[np.full(2*track.n, np.inf), track.W[:,1], np.full(2*track.n, np.inf), np.full(track.n , p.E_max)],
        lbg = np.r_[np.full(2*(track.n-1), -np.inf), np.zeros(track.n-1), np.full(2*track.n, -np.inf), np.zeros(track.n), np.zeros(track.n-1), 0.0, 0.0],
        ubg = np.r_[np.zeros(2*(track.n-1)), np.zeros(track.n-1), np.zeros(2*track.n), np.full(track.n, np.inf), np.zeros(track.n-1), 0.0, 0.0])

    stats = solver.stats()

    print(stats['return_status']) 
    
    return solution


def plot_results(track, solution):
    """Plots the optimal lap's caracteristics given the solution of the optimisation"""

    laptime = float(solution["f"])
    Traj = position(track, solution["x"])
    D = distance(track, solution["x"])
    V = speed(solution["x"], track.n)
    A = acceleration(track, solution["x"])

    x = unpack(solution["x"], track.n)
    T_ice = x["T_ice"]
    T_mguk = x["T_mguk"]
    E = x["E"]

    layout = [["A", "B"], 
            ["A", "C"],
            ["D", "E"]]
        
    fig, ax = plt.subplot_mosaic(layout, layout="constrained")

    fig.suptitle(f"Optimal 1-Lap Simulation – {track.name} | Lap Time = {int(laptime//60)}:{laptime%60:06.3f}", fontsize=14, fontweight="bold")

    ax["D"].plot(D, T_ice, label = "ICE")
    ax["D"].plot(D, T_mguk, color = "red", label = "MGU-K")
    ax["E"].plot(D, E/1e6)

    track.trace_track(ax["A"])
    ax["A"].plot(Traj[:,0], Traj[:,1], color = "red", lw = 0.7)
    ax["B"].plot(D, V*3.6)
    ax["C"].plot(D, A/p.g, color = "red")

    for key in ["B", "C", "D", "E"]:
        ax[key].minorticks_on()
        ax[key].grid(True, which="major", alpha=0.35, linewidth=0.8)
        ax[key].grid(True, which="minor", alpha=0.15, linestyle="--", linewidth=0.5)

    ax["A"].set_title("Optimal Racing Line")
    ax["B"].set_title("Speed Profile")
    ax["C"].set_title("Total Acceleration")
    ax["D"].set_title("ICE & MGU-K Wheel Torque")
    ax["E"].set_title("Battery Energy")

    ax["B"].set_xlabel("Distance (m)")
    ax["B"].set_ylabel("Speed (km/h)")
    ax["C"].set_xlabel("Distance (m)")
    ax["C"].set_ylabel("Acceleration (g)")
    ax["D"].set_xlabel("Distance (m)")
    ax["D"].set_ylabel("Wheel Torque (N.m)")
    ax["E"].set_xlabel("Distance (m)")
    ax["E"].set_ylabel("Energy level (MJ)")

    ax["D"].legend()

    plt.show()


def distance(track, v):
    d = dist_int(track, v)
    return np.r_[0, np.cumsum(d)]


def acceleration(track, v):
    V = speed(v, track.n)
    dt = time_int(track, v)
    A = (V[1:] - V[0:-1])/dt
    T = heading_change(v, track.n)
    return np.r_[0, np.reshape(np.sqrt(A**2 + ((V[1:]+V[0:-1])/2)**2*(T/dt)**2), track.n-1)]


if __name__ == "__main__":

    austin = Track("Austin")
    sol = solve_lap(austin)
    plot_results(austin, sol)
