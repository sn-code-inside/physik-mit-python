﻿"""Vergleich einer simulierten Resonanzkurve mit der
analytischen Lösung. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Simulationsdauer [s].
T = 60

# Federkonstante [N/m].
D = 400.0

# Masse [kg].
m = 1e-3

# Reibungskoeffizient [kg/s].
b = 0.1

# Anregungsamplitude [m].
A = 1e-3

# Parameter für die Frequenzmodulation:
# Minimale und maximale Kreisfrequenz der Anregung [1/s].
omega_min = 300
omega_max = 1000

# Mittenkreisfrequenz [1/s] und Kreisfrequenzhub [1/s].
omega_0 = (omega_max + omega_min) / 2
omega_hub = (omega_max - omega_min) / 2

# Kreisfrequenz der Modulation [s].
omega_mod = 2 * 2 * np.pi / T


def omega_a(t):
    """Anregungskreisfreuquenz als Funktion der Zeit. """
    return omega_0 - omega_hub * np.cos(omega_mod * t)


def x_a(t):
    """Anregungsfunktion. """
    phi = omega_0 * t - (
            omega_hub / omega_mod * np.sin(omega_mod * t))
    return A * np.sin(phi)


def dgl(t, u):
    x, v = np.split(u, 2)
    F = - D * (x - x_a(t)) - b * v
    return np.concatenate([v, F / m])


def umkehrpunkt(t, u):
    """Ereignisfunktion: Maxima/Minima der Schwingung. """
    y, v = u
    return v


# Zustandsvektor für den Anfangszustand.
u0 = np.array([0, 0])

# Löse die Differentialgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   events=umkehrpunkt,
                                   dense_output=True)

# Bestimme die Zeitpunkte der Umkehrpunkte.
t = result.t_events[0]

# Bestimme für diese Zeitpunkte den Betrag x der Auslenkung.
x, v = result.sol(t)
x = np.abs(x)

# Berechne die jeweils aktuelle Anregungskreisfrequenz.
omega = omega_a(t)

# Berechne die analytische Amplitudenresonanzkurve.
omega0 = np.sqrt(D / m)
delta = b / (2 * m)
x_analyt = A * omega0 ** 2 / np.abs(
    omega0 ** 2 - omega ** 2 + 2 * 1j * delta * omega)

# Plotte beide Ergebnisse.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$\\omega$ [1/s]')
ax.set_ylabel('Amplitude [m]')
ax.set_xlim(omega_min, omega_max)
ax.grid()
ax.plot(omega, x, 'b-', label='Simulation')
ax.plot(omega, x_analyt, 'r-', label='Analytisch')
ax.legend()

# Zeige die Grafik an.
plt.show()

