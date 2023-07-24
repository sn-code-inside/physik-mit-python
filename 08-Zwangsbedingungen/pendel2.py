"""Simulation eines ebenen Pendels mit Stabilisierung.
Formalerer Ansatz über den Gradienten der Zwangsbedingung. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeitdauer T und dargestellte Schrittweite dt [s].
T = 20
dt = 0.02

# Masse des Körpers [kg].
m = 1.0

# Länge des Pendels [m].
L = 0.7

# Anfangsauslenkung [rad].
phi0 = math.radians(20.0)

# Erdbeschleunigung [m/s²].
g = 9.81

# Anfangsort [m].
r0 = L * np.array([math.sin(phi0), -math.cos(phi0)])

# Anfangsgeschwindigkeit [m/s].
v0 = np.array([0, 0])

# Vektor der Gewichtskraft.
F_g = m * g * np.array([0, -1])

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
    return np.array([[2.0, 0.0], [0.0, 2.0]])


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

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.set_ylim([-0.95, 0.05])
ax.grid()

# Plotte die Bahnkurve.
bahn, = ax.plot(r[0], r[1], '-b')

# Erzeuge eine Punktplot, für die Position des Pendelkörpers
# und einen Linienplot für die Stange.
koerper, = ax.plot([0], [0], 'o', color='red', markersize=10,
                   zorder=5)
stange, = ax.plot([0, 0], [0, 0], '-', color='black')


def update(n):
    # Aktualisiere die Bahnkurve.
    bahn.set_data(r[0, :n + 1], r[1, :n + 1])

    # Aktualisiere die Position des Pendelkörpers.
    koerper.set_data(r[:, n])

    # Aktualisiere die Pendelstange.
    stange.set_data([0, r[0, n]], [0, r[1, n]])

    return stange, koerper, bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
