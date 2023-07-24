"""Simulation eines freien Falls mit Berücksichtigung der
Corioliskraft. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# Erdbeschleunigung [m/s²].
g = 9.81

# Anfangsort [m].
r0 = np.array([0, 0, 100.0])

# Berechne den Vektor der Anfangsgeschwindigkeit [m/s].
v0 = np.array([0.0, 0.0, 0.0])

# Breitengrad [rad].
theta = math.radians(49.4)

# Vektor der Winkelgeschwindigkeit [rad/s].
omega = 7.292e-5 * np.array([0, math.cos(theta), math.sin(theta)])


def dgl(t, u):
    r, v = np.split(u, 2)

    # Schwerebescheleunigung.
    a_g = g * np.array([0, 0, -1])

    # Coriolisbeschleunigung.
    a_c = - 2 * np.cross(omega, v)

    # Summe aus Schwere- und Coriolisbeschleunigung.
    a = a_g + a_c
    return np.concatenate([v, a])


def aufprall(t, u):
    """Ereignisfunktion: Liefert einen Vorzeichenwechsel beim
    Auftreffen auf dem Erdboden (z=0). """
    r, v = np.split(u, 2)
    return r[2]


# Beende die Simulation soll beim Auftreten des Ereignisses.
aufprall.terminal = True

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   events=aufprall,
                                   dense_output=True)

# Berechne die Interpolation auf einem feinen Raster.
t = np.linspace(0, np.max(result.t), 1000)
r, v = np.split(result.sol(t), 2)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 3))
fig.set_tight_layout(True)

# Plotte die Bahnkurve in der Seitenansicht.
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('x [mm]')
ax1.grid()
ax1.plot(t, 1e3 * r[0], '-b')

# Plotte die Bahkurve in der Aufsicht.
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y [mm]')
ax2.grid()
ax2.plot(t, 1e3 * r[1], '-b')

# Plotte die Bahkurve in der Aufsicht.
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('z [m]')
ax3.grid()
ax3.plot(t, r[2], '-b')

# Zeige die Grafik an.
plt.show()
