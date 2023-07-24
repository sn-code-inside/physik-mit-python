"""Simulation des Botafumeiro. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Simulationsdauer T und Zeitschrittweite dt [s].
T = 1000
dt = 0.5

# Masse des Pendelkörpers [kg].
m = 50.0

# Betrag der Erdbeschleunigung [m/s²].
g = 9.81

# Anfangslänge des Pendels [m].
l0 = 30.0

# Amplitude der Längenvariation [m].
l1 = 0.2

# Kreisfrequenz der Anregung.
omega = 2 * np.sqrt(g / l0)

# Anfangsauslenkung [rad].
phi0 = math.radians(1.0)

# Parameter für die Baumgarte-Stabilisierung [1/s].
alpha = 10.0
beta = alpha

# Angansposition [m].
r0 = l0 * np.array([math.sin(phi0), -math.cos(phi0)])

# Anfangsgeschwindigkeit [m/s].
v0 = np.array([0.0, 0.0])

# Vektor der Gewichtskraft.
Fg = m * g * np.array([0, -1])


def L(t):
    """Pendellänge als Funktion der Zeit. """
    return l0 + l1 * np.sin(omega * t)


def dt_L(t):
    """Zeitableitung dL/dt. """
    return l1 * omega * np.cos(omega * t)


def d2t_L(t):
    """Zweite Zeitableitung d²L/dt². """
    return -l1 * omega**2 * np.sin(omega * t)


def h(t, r):
    """Zwangsbedingung h(r). """
    return r @ r - L(t)**2


def dt_h(t, r):
    """Partielle Zeitableitung der Zwangsbedingung. """
    return - 2 * L(t) * dt_L(t)


def d2t_h(t, r):
    """zweite partielle Zeitableitung der Zwangsbedingung. """
    return -2 * dt_L(t)**2 - 2 * L(t) * d2t_L(t)


def grad_h(t, r):
    """Gradient dh / dx_i """
    return 2 * r


def dtgrad_h(t, r):
    """Partielle Zeitableitung des Gradienten. """
    return np.zeros_like(r)


def hesse_h(t, r):
    """Hesse-Matrix H[i, j] =  d²h / (dx_i dx_j) """
    return np.array([[2.0, 0.0], [0.0, 2.0]])


def dgl(t, u):
    r, v = np.split(u, 2)

    # Gewichtskraft.
    F_g = np.array([0, -m*g])

    # Stelle die Gleichungen für die lambdas auf.
    grad = grad_h(t, r)
    hesse = hesse_h(t, r)

    # Berechne F unter Berücksichtigung der zusätzlichen Terme bei
    # explizit zeitabhängigen Zwangsbedingungen.
    F = - v @ hesse @ v - grad @ (F_g / m)
    F += - 2 * alpha * grad @ v - beta ** 2 * h(t, r)
    F += -2 * alpha * dt_h(t, r) - dtgrad_h(t, r) @ v - d2t_h(t, r)

    # Berechne G und bestimme lambda.
    G = (grad / m) @ grad
    lam = F / G

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_g + lam * grad) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-6,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('t [s]')
ax.set_ylabel('y [m]')
ax.grid()

# Plotte die x-Koordinate des Pendels als Funktion der Zeit.
ax.plot(t, r[0])

# Zeige die Grafik an.
plt.show()
