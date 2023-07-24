"""Simulation eines Fadenpendels mit Stabilisierung. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeitdauer T und dargestellte Schrittweite dt [s].
T = 20
dt = 0.005

# Masse des Körpers [kg].
m = 1.0

# Länge des Pendels [m].
L = 0.7

# Toleranz zur Erkennung des Durchhängens [m].
epsilon = 0.001

# Anfangsgeschwindigkeit im tiefsten Punkt [m/s].
v0 = 5.7

# Erdbeschleunigung [m/s²].
g = 9.81

# Angansposition [m].
r0 = np.array([0.0, -L])

# Anfangsgeschwindigkeit [m/s].
v0 = np.array([v0, 0])

# Vektor der Gewichtskraft.
Fg = m * g * np.array([0, -1])

# Parameter für die Baumgarte-Stabilisierung [1/s].
alpha = 200
beta = alpha


def h(r):
    """Zwangsbedingung h(r). """
    return r @ r - L ** 2


def grad_h(r):
    """Gradient g[i] =  dh / dx_i """
    return 2 * r


def hesse_h(r):
    """Hesse-Matrix H[i, j] =  d²h / (dx_i dx_j) """
    return 2 * np.eye(2)


def dgl(t, u):
    r, v = np.split(u, 2)

    # Gewichtskraft.
    F_g = np.array([0, -m*g])

    # Stelle die Gleichungen für die lambdas auf.
    grad = grad_h(r)
    hesse = hesse_h(r)
    F = - v @ hesse @ v - grad @ (F_g / m)
    F += - 2 * alpha * grad @ v - beta ** 2 * h(r)
    G = (grad / m) @ grad

    # Berechne die lambda.
    lam = F / G

    # Es tritt keine Zwangskraft auf, wenn diese nach außen
    # gerichtet wäre:
    lam = min(lam, 0)

    # Es tritt keine Zwangskraft auf, wenn das Seil durchhängt.
    if r @ r < (L - epsilon)**2:
        lam = 0

    # Wenn das Seil wieder straff ist und es eine
    # Geschwindigkeitskomponente nach außen gibt, dann wird
    # diese auf Null gesetzt.
    if (r @ r > (L + epsilon) ** 2) and (grad @ v > 0):
        v -= (grad @ v) * grad / (grad @ grad)

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_g + lam * grad) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   rtol=1e-6,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.05 * L, 1.05 * L])
ax.set_ylim([-1.05 * L, 1.05 * L])
ax.set_aspect('equal')
ax.grid()

# Stelle den Kreis dar, der die Zwangsbedingung repäsentiert.
c = mpl.patches.Circle([0, 0], L, fill=False, zorder=2)
ax.add_artist(c)

# Erzeuge eine Punktplot, für die Position des Pendelkörpers,
# einen Linienplot für die Stange und einen Linienplot für die
# Bahnkurve.
pendel, = ax.plot([0], [0], 'o', color='red', markersize=10,
                  zorder=5)
stange, = ax.plot([0, 0], [0, 0], '-', color='black',
                  zorder=4)
bahn, = ax.plot([0], [0], '-b', zorder=3)


def update(n):
    # Aktualisiere die Position des Pendels.
    stange.set_data([0, r[0, n]], [0, r[1, n]])
    pendel.set_data(r[:, n])

    # Stelle die Bahnkurve bis zum aktuellen Zeitpunkt dar.
    bahn.set_data(r[:, :n+1])

    # Färbe die Pendelstange hellgrau, wenn das Pendel durchhängt.
    if np.linalg.norm(r[:, n]) < L - epsilon:
        stange.set_color('lightgray')
    else:
        stange.set_color('black')

    return stange, pendel, bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
