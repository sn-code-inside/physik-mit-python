"""Simulation des Vierfachpendels. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Raumdimension dim, Anzahl der Teilchen N und
# Anzahl der Zwangsbedingungen R.
dim = 2
N = 4
R = 4

# Massen der Pendelkörpers [kg].
m1 = 1.0
m2 = 1.0
m3 = 1.0
m4 = 1.0

# Länge der Pendelstangen [m].
l1 = 0.8
l2 = 0.4
l3 = 0.2
l4 = 0.1

# Simulationsdauer T und Zeitschrittweite dt [s].
T = 10
dt = 0.002

# Anfangsauslenkung [rad].
phi1 = math.radians(130.0)
phi2 = math.radians(0.0)
phi3 = math.radians(0.0)
phi4 = math.radians(0.0)

# Vektor der Anfangsposition [m].
r01 = l1 * np.array([math.sin(phi1), -math.cos(phi1)])
r02 = r01 + l2 * np.array([math.sin(phi2), -math.cos(phi2)])
r03 = r02 + l3 * np.array([math.sin(phi3), -math.cos(phi3)])
r04 = r03 + l4 * np.array([math.sin(phi4), -math.cos(phi4)])

# Array mit den Komponenten der Anfangspositionen [m].
r0 = np.concatenate((r01, r02, r03, r04))

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
v0 = np.zeros(N * dim)

# Array der Masse für jede Komponente [kg].
m = np.array([m1, m1, m2, m2, m3, m3, m4, m4])

# Betrag der Erdbeschleunigung [m/s²].
g = 9.81

# Parameter für die Baumgarte-Stabilisierung [1/s].
alpha = 10.0
beta = alpha


def h(r):
    """Zwangsbedingungen h_a(r). """
    r = np.split(r, N)
    d1 = r[0]
    d2 = r[1] - r[0]
    d3 = r[2] - r[1]
    d4 = r[3] - r[2]
    return np.array([d1 @ d1 - l1 ** 2,
                     d2 @ d2 - l2 ** 2,
                     d3 @ d3 - l3 ** 2,
                     d4 @ d4 - l4 ** 2])


def grad_h(r):
    """Gradient g der Zwangsbedingungen.
        g[a, i] =  dh_a / dx_i
    """
    r = np.split(r, N)
    g = np.zeros((R, N, dim))

    # Erste Zwangsbedingung.
    g[0, 0] = 2 * r[0]

    # Zweite Zwangsbedingung.
    g[1, 0] = 2 * (r[0] - r[1])
    g[1, 1] = 2 * (r[1] - r[0])

    # Dritte Zwangsbedingung.
    g[2, 1] = 2 * (r[1] - r[2])
    g[2, 2] = 2 * (r[2] - r[1])

    # Vierte Zwangsbedingung.
    g[3, 2] = 2 * (r[2] - r[3])
    g[3, 3] = 2 * (r[3] - r[2])

    return g.reshape(R, N * dim)


def hesse_h(r):
    """Hesse-Matrix H der Zwangsbedingungen.
        H[a, i, j] =  d²h_a / (dx_i dx_j)
    """
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

    # Vierte Zwangsbedingung.
    h[3, 2, :, 2, :] = 2 * E
    h[3, 2, :, 3, :] = -2 * E
    h[3, 3, :, 2, :] = -2 * E
    h[3, 3, :, 3, :] = 2 * E

    return h.reshape(R, N * dim, N * dim)


def dgl(t, u):
    r, v = np.split(u, 2)

    # Lege die externe Kraft fest.
    F_ext = m * np.array([0, -g, 0, -g, 0, -g, 0, -g])

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


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-6,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Zerlege den Ors- und Geschwindigkeitsvektor in die
# entsprechenden Vektoren für die drei Massen.
r1, r2, r3, r4 = np.split(r, 4)
v1, v2, v3, v4 = np.split(v, 4)

# Berechne die tatsächliche Pendellänge für jeden Zeitpunkt.
len1 = np.linalg.norm(r1, axis=0)
len2 = np.linalg.norm(r1-r2, axis=0)
len3 = np.linalg.norm(r2-r3, axis=0)
len4 = np.linalg.norm(r3-r4, axis=0)

# Berechne die Gesamtenergie für jeden Zeitpunkt.
E_pot = (m1 * g * r1[1, :] + m2 * g * r2[1, :] +
         m3 * g * r3[1, :] + m4 * g * r4[1, :])
E_kin = 0
E_kin += 0.5 * m1 * np.sum(v1**2, axis=0)
E_kin += 0.5 * m2 * np.sum(v2**2, axis=0)
E_kin += 0.5 * m3 * np.sum(v3**2, axis=0)
E_kin += 0.5 * m4 * np.sum(v4**2, axis=0)
E = E_kin + E_pot

# Gib eine Tabelle der Minimal- und Maximalwerte aus.
print(f'      minimal        maximal')
print(f'  l1: {np.min(len1):.7f} m    {np.max(len1):.7f} m')
print(f'  l2: {np.min(len2):.7f} m    {np.max(len2):.7f} m')
print(f'  l3: {np.min(len3):.7f} m    {np.max(len3):.7f} m')
print(f'  l4: {np.min(len4):.7f} m    {np.max(len4):.7f} m')
print(f'   E: {np.min(E):.7f} J    {np.max(E):.7f} J')

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 0.4])
ax.set_aspect('equal')
ax.grid()

# Erzeuge je einen Punktplot, für die Position des Massen.
p1, = ax.plot([0], [0], 'bo', markersize=8, zorder=5)
p2, = ax.plot([0], [0], 'ro', markersize=8, zorder=5)
p3, = ax.plot([0], [0], 'go', markersize=8, zorder=5)
p4, = ax.plot([0], [0], 'mo', markersize=8, zorder=5)
lines, = ax.plot([0, 0], [0, 0], 'k-', zorder=4)


def update(n):
    # Aktualisiere die Position der Pendelkörper.
    p1.set_data(r1[:, n])
    p2.set_data(r2[:, n])
    p3.set_data(r3[:, n])
    p4.set_data(r4[:, n])

    # Aktualisiere die Position der Pendelstangen.
    p0 = np.array((0, 0))
    points = np.array([p0,
                       r1[:, n], r2[:, n], r3[:, n], r4[:, n]])
    lines.set_data(points.T)

    return p1, p2, p3, p4, lines


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
