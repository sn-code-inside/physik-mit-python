"""Simulation eines Kanonenschusses mit Berücksichtigung der
Corioliskraft. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# Masse des Körpers [kg].
m = 14.5

# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 8e-2 ** 2

# Erdbeschleunigung [m/s²].
g = 9.81

# Luftdichte [kg/m³].
rho = 1.225

# Anfangsort [m].
r0 = np.array([0, 0, 10.0])

# Abwurfwinkel [rad].
alpha = math.radians(42.0)

# Mündungsgeschwindigkeit [m/s].
v0 = 150.0

# Berechne den Vektor der Anfangsgeschwindigkeit [m/s].
v0 = np.array([v0 * math.cos(alpha), 0, v0 * math.sin(alpha)])

# Breitengrad [rad].
theta = math.radians(49.4)

# Vektor der Winkelgeschwindigkeit [rad/s].
omega = 7.292e-5 * np.array([0, math.cos(theta), math.sin(theta)])


def F(r, v):
    """Vektor der Kraft als Funktion von Ort r und
    Geschwindigkeit v. """
    Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
    Fg = m * g * np.array([0, 0, -1])
    Fc = - 2 * m * np.cross(omega, v)
    return Fg + Fr + Fc


def dgl(t, u):
    r, v = np.split(u, 2)
    return np.concatenate([v, F(r, v) / m])


def aufprall(t, u):
    """Ereignisfunktion: liefert einen Vorzeichenwechsel beim
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
t_s = result.t
r_s, v_s = np.split(result.y, 2)

# Berechne die Interpolation auf einem feinen Raster.
t = np.linspace(0, np.max(t_s), 1000)
r, v = np.split(result.sol(t), 2)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 6))
fig.set_tight_layout(True)

# Plotte die Bahnkurve in der Seitenansicht.
ax1 = fig.add_subplot(2, 1, 1)
ax1.tick_params(labelbottom=False)
ax1.set_ylabel('z [m]')
ax1.set_aspect('equal')
ax1.grid()
ax1.plot(r_s[0], r_s[2], '.b')
ax1.plot(r[0], r[2], '-b')

# Plotte die Bahkurve in der Aufsicht.
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
ax2.grid()
ax2.plot(r_s[0], r_s[1], '.b')
ax2.plot(r[0], r[1], '-b')

# Zeige die Grafik an.
plt.show()
