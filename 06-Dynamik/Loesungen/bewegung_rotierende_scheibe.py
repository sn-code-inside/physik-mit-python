﻿"""Simulation der Bewegung einer Masse auf einer rotierenden
Scheibe im Bezugssystem der rotierenden Scheibe. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import scipy.interpolate

# Simulationsdauer T und dargestellte Zeitschrittweite dt [s].
T = 5.0
dt = 0.005

# Anfangsort [m].
r0 = np.array([0.1, 0, 0])

# Anfangsgeschwindigkeit [m/s].
v0 = np.array([0.0, 0.0, 0.0])

# Drehzahl der Scheibe [1/s].
f = 1

# Vektor der Winkelgeschwindigkeit [1/s].
omega = np.array([0, 0, 2 * math.pi * f])


def dgl(t, u):
    r, v = np.split(u, 2)

    # Berechne die Coriolisbeschleunigung.
    a_c = - 2 * np.cross(omega, v)

    # Berechne die Zentrifugalbeschleunigung.
    a_z = - np.cross(omega, np.cross(omega, r))

    # Berechne die Gesamtbeschleunigung.
    a = a_c + a_z

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid()

# Plotte die Bahnkurve in der Aufsicht.
bahn, = ax.plot(r[0], r[1], '-b', zorder=3)

# Erzeuge eine Punktplot, für die Position des Körpers.
koerper, = ax.plot([0], [0], 'o', zorder=5,
                   color='red', markersize=10)


def update(n):
    # Aktualisiere die Position des Körpers.
    koerper.set_data(r[0:2, n].reshape(-1, 1))
    # Plotte die Bahnkurve bis zum aktuellen Zeitpunkt.
    bahn.set_data(r[0:2, :n+1])

    return koerper, bahn


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
