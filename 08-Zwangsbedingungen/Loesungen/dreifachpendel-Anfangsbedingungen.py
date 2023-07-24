"""Simultane Simulation von zwei Dreifachpendeln mit leicht
unterschiedlichen Anfangsbedingungen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Raumdimension dim, Anzahl der Teilchen N und
# Anzahl der Zwangsbedingungen R.
dim = 2
N = 3
R = 3

# Massen der Pendelkörpers [kg].
m1 = 1.0
m2 = 1.0
m3 = 1.0

# Länge der Pendelstangen [m].
l1 = 0.6
l2 = 0.3
l3 = 0.15

# Simulationsdauer T und Zeitschrittweite dt [s].
T = 10
dt = 0.002

# Anfangsauslenkungen für die beiden Simulationsläufe [rad].
phi1 = np.radians(np.array([130.0, 130.001]))
phi2 = np.radians(np.array([0.0, 0.0]))
phi3 = np.radians(np.array([0.0, 0.0]))

# Array der Anfangspositionen [m].
r01 = l1 * np.array([np.sin(phi1), -np.cos(phi1)])
r02 = r01 + l2 * np.array([np.sin(phi2), -np.cos(phi2)])
r03 = r02 + l3 * np.array([np.sin(phi3), -np.cos(phi3)])

# Array mit den Komponenten der Anfangspositionen [m].
r0 = np.concatenate((r01, r02, r03))

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
v0 = np.zeros((N * dim, 2))

# Array der Masse für jede Komponente [kg].
m = np.array([m1, m1, m2, m2, m3, m3])

# Betrag der Erdbeschleunigung [m/s²].
g = 9.81

# Parameter für die Baumgarte-Stabilisierung [1/s].
alpha = 10.0
beta = alpha


def h(r):
    """Zwangsbedingungen h_a(r). """
    r = r.reshape(N, dim)
    d1 = r[0]
    d2 = r[1] - r[0]
    d3 = r[2] - r[1]
    return np.array([d1 @ d1 - l1 ** 2,
                     d2 @ d2 - l2 ** 2,
                     d3 @ d3 - l3 ** 2])


def grad_h(r):
    """Gradient g der Zwangsbedingungen.
        g[a, i] =  dh_a / dx_i """
    r = r.reshape(N, dim)
    g = np.zeros((R, N, dim))

    # Erste Zwangsbedingung.
    g[0, 0] = 2 * r[0]

    # Zweite Zwangsbedingung.
    g[1, 0] = 2 * (r[0] - r[1])
    g[1, 1] = 2 * (r[1] - r[0])

    # Dritte Zwangsbedingung.
    g[2, 1] = 2 * (r[1] - r[2])
    g[2, 2] = 2 * (r[2] - r[1])

    return g.reshape(R, N * dim)


def hesse_h(r):
    """Hesse-Matrix H der Zwangsbedingungen.
        H[a, i, j] =  d²h_a / (dx_i dx_j) """
    h = np.zeros((R, N, dim, N, dim))

    # Erstelle eine dim x dim - Einheitsmatrix.
    E = np.eye(dim)

    # Erste Zwangsbedingung.
    h[0, 0, :, 0, :] = 2 * E

    # Zweite Zwangsbedingung.
    h[1, 0, :, 0, :] = 2 * E
    h[1, 0, :, 1, :] = -2 * E
    h[1, 1, :, 0, :] = -2 * E
    h[1, 1, :, 1, :] = 2 * E

    # Dritte Zwangsbedingung.
    h[2, 1, :, 1, :] = 2 * E
    h[2, 1, :, 2, :] = -2 * E
    h[2, 2, :, 1, :] = -2 * E
    h[2, 2, :, 2, :] = 2 * E

    return h.reshape(R, N * dim, N * dim)


def dgl(t, u):
    r, v = np.split(u, 2)

    # Lege die externe Kraft fest.
    F_ext = m * np.array([0, -g, 0, -g, 0, -g])

    # Stelle die Gleichungen für die lambdas auf.
    grad = grad_h(r)
    hesse = hesse_h(r)
    F = - v @ hesse @ v - grad @ (F_ext / m)
    F += - 2 * alpha * grad @ v - beta ** 2 * h(r)
    G = (grad / m) @ grad.T

    # Berechne die lambdas.
    lam = np.linalg.solve(G, F)

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_ext + lam @ grad) / m

    return np.concatenate([v, a])


# Lege die Zustandsvektoren zum Zeitpunkt t=0 fest.
u0array = np.concatenate((r0, v0))

# Erzeuge Arrays für die Ergebnisse für beide Simulationsläufe
t = np.arange(0, T, dt)
r = np.zeros((u0array.shape[1], N, dim, t.size))
v = np.zeros((u0array.shape[1], N, dim, t.size))

# Führe für jeden Satz von Anfangsbedingungen eine Simulation
# durch.
for i, u0 in enumerate(u0array.T):
    result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                       rtol=1e-6,
                                       t_eval=t)
    t = result.t
    r[i], v[i] = np.reshape(result.y, (2, N, dim, -1))


# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 0.4])
ax.set_aspect('equal')
ax.grid()

# Erzeuge je einen Punktplot, für die Position des Massen.
p1, = ax.plot([0], [0], 'bo', markersize=8, zorder=5)
p2, = ax.plot([0], [0], 'ro', markersize=8, zorder=5)
p3, = ax.plot([0], [0], 'go', markersize=8, zorder=5)

# Erzeuge je einen Linienplot für die Darstellung der
# Pendelstangen.
line1, = ax.plot([0, 0], [0, 0], 'k-', zorder=4)
line2, = ax.plot([0, 0], [0, 0], 'k-', zorder=4)

# Erzeuge ein Textfeld für die Angabe der Zeit.
text = ax.text(-1.0, 0.3, '')


def update(n):
    # Aktualisiere die Position der Pendelkörper.
    p1.set_data(r[:, 0, :, n].T)
    p2.set_data(r[:, 1, :, n].T)
    p3.set_data(r[:, 2, :, n].T)

    # Aktualisiere die Position der Pendelstangen der ersten
    # Simulation.
    points = np.zeros((2, 4))
    points[:, 1:] = r[0, :, :, n].T
    line1.set_data(points)

    # Aktualisiere die Position der Pendelstangen der zweiten
    # Simulation.
    points = np.zeros((2, 4))
    points[:, 1:] = r[1, :, :, n].T
    line2.set_data(points)

    # Aktualisiere den Text des Textfeldes.
    text.set_text(f't = {t[n]:5.2f}')

    return p1, p2, p3, line1, line2, text


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
