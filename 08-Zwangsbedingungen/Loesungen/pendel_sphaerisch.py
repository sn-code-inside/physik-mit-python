"""Simulation eines spärischen Pendels. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import mpl_toolkits.mplot3d

# Simulationsdauer und Schrittweite [s].
T = 100.0
dt = 0.02

# Masse des Körpers [kg].
m = 1.0

# Länge des Pendels [m].
L = 0.7

# Anfangsauslenkung [rad].
phi0 = math.radians(40.0)

# Anfangsgeschwindigkeit [m/s].
v0 = 0.4

# Vektor der Anfangsposition [m].
r0 = np.array([L * math.sin(phi0), 0, -L * math.cos(phi0)])

# Vektor der Anfangsgeschwindigkeit [m/s].
v0 = np.array([0, v0, 0])

# Erdbeschleunigung [m/s²].
g = 9.81

# Vektor der Gewichtskraft.
F_g = np.array([0, 0, -m * g])

# Parameter für die Baumgarte-Stabilisierung.
beta = alpha = 10.0


def h(r):
    """Zwangsbedingung: h(r) """
    return r @ r - L ** 2


def grad_h(r):
    """Gradient: g[i] =  dh / dx_i """
    return 2 * r


def hesse_h(r):
    """Hesse-Matrix: H[i, j] =  d²h / (dx_i dx_j) """
    return 2 * np.eye(3)


def dgl(t, u):
    r, v = np.split(u, 2)

    # Stelle die Gleichungen für lambda auf.
    grad = grad_h(r)
    hesse = hesse_h(r)
    F = - v @ hesse @ v - grad @ (F_g / m)
    F += - 2 * alpha * grad @ v - beta ** 2 * h(r)
    G = (grad / m) @ grad

    # Berechne lambda.
    lam = F / G

    # Berechne die Zwangskraft.
    F_z = lam * grad

    # Berechne den Vektor der Beschleunigung.
    a = (F_z + F_g) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1, projection='3d', elev=50, azim=45)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.grid()

# Plotte die Bahnkurve.
bahn, = ax.plot(r[0], r[1], r[2], '-b', lw=0.4)

# Erzeuge eine Punktplot, für die Position des Pendelkörpers und
# einen Linienplot für die Pendelstange.
pendel, = ax.plot([0], [0], 'o', color='red',
                  markersize=10, zorder=5)
stange, = ax.plot([0, 0], [0, 0], [0, -L], '-', color='black')


def update(n):
    # Plotte die bisher zurückgelegte Bahnkurve.
    bahn.set_data(r[0, :n+1], r[1, :n+1])
    bahn.set_3d_properties(r[2, :n+1])

    # Aktualisiere die Position des Pendelkörpers.
    pendel.set_data(np.array([r[0, n]]), np.array([r[1, n]]))
    pendel.set_3d_properties(np.array([r[2, n]]))

    # Aktualisiere die Abbildung der Pendelstange
    stange.set_data(np.array([0, r[0, n]]),
                    np.array([0, r[1, n]]))
    stange.set_3d_properties(np.array([0, r[2, n]]))
    return pendel, stange, bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30)
plt.show()
