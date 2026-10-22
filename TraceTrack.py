import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def normal(track) : # Returns the normal vector to the middle line in each point of the track, needed to be able to consider the width

    db = pd.read_csv(f"tracks/{track}.csv")
    X = np.append(db[["# x_m", "y_m"]].to_numpy(), np.reshape(db[["# x_m", "y_m"]].to_numpy()[0], (1,2)), axis = 0)      #closes the lap
    dX = np.diff(X, axis = 0)

    norm = np.sqrt(dX[:,0]**2 + dX[:,1]**2)
    n = len(X)

    N = np.append(-np.reshape(dX[:,1]/norm,(n-1,1)), np.reshape(dX[:,0]/norm,(n-1,1)), axis = 1)

    return np.append(N, np.reshape(N[0], (1,2)), axis = 0)

def TraceTrack(track, ax) :                                    # Traces the track limits 
    Track = pd.read_csv(f"tracks/{track}.csv").to_numpy()
    db = pd.read_csv(f"tracks/{track}.csv")
    X_m = np.append(Track[:,0], Track[0,0])
    Y_m = np.append(Track[:,1], Track[0,1])
    W = np.append(db[["w_tr_right_m","w_tr_left_m"]].to_numpy(), np.reshape(db[["w_tr_right_m","w_tr_left_m"]].to_numpy()[0], (1,2)), axis = 0)
    N = normal(track)

    X_l = X_m + N[:,0]*W[:,1]
    Y_l = Y_m + N[:,1]*W[:,1]
    X_r = X_m - N[:,0]*W[:,0]
    Y_r = Y_m - N[:,1]*W[:,0]
    ax.plot(X_l, Y_l, color="black", lw=0.4)
    ax.plot(X_r, Y_r, color="black", lw=0.4)
    ax.axis("equal")

def TraceRaceline(track, ax) :                                  # Traces the Raceline
    Raceline = pd.read_csv(f"racelines/{track}.csv").to_numpy()
    X = Raceline[:,0]
    Y = Raceline[:,1]
    ax.plot(np.append(X, X[0]), np.append(Y, Y[0]), color="red", lw=0.4)
    ax.axis("equal")
