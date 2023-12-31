﻿"""Demonstration einer Resonanzkurve mit Hysterese bei einer
degressiven Feder. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.integrate
import matplotlib.animation

# Simulationsdauer [s].
T = 60

# Anfängliche Federhärte [N/m].
D0 = 2000.0

# Masse [kg].
m = 1e-3

# Reibungskoeffizient [kg/s].
b = 0.1

# Konstante für den nichtlinearen Term [m].
alpha = -2e-3

# Anregungsamplitude [m].
A = 1e-3

# Parameter für die Frequenzmodulation:
# Minimale und maximale Kreisfrequenz der Anregung [1/s].
omega_min = 400
omega_max = 1400

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


def Federkraft(x):
    return -D0 * alpha * np.expm1(np.abs(x)/alpha) * np.sign(x)


def dgl(t, u):
    x, v = np.split(u, 2)
    F = Federkraft(x - x_a(t)) - b * v
    return np.concatenate([v, F / m])


def umkehrpunkt(t, u):
    """Ereignisfunktion: Maxima/Minima der Schwingung. """
    y, v = u
    return v


# Zustandsvektor für den Anfangszustand.
u0 = np.array([0, 0])

# Löse die Differentialgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-4,
                                   events=umkehrpunkt,
                                   dense_output=True)

# Bestimme die Zeitpunkte der Umkehrpunkte.
t = result.t_events[0]

# Bestimme für diese Zeitpunkte den Betrag x der Auslenkung.
x, v = result.sol(t)
x = np.abs(x)

# Berechne die jeweils aktuelle Anregungskreisfrequenz.
omega = omega_a(t)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('$\\omega$ [1/s]')
ax.set_ylabel('Amplitude [m]')
ax.set_xlim(omega_min, omega_max)
ax.grid()

# Erzeuge eine Linienplot und einen Punktplot.
plot, = ax.plot(omega, x, '-', zorder=4)
punkt, = ax.plot([0], [0], 'or', zorder=5)


def update(n):
    punkt.set_data([omega[n]], [x[n]])
    plot.set_data(omega[0:n+1], x[0:n+1])
    return punkt, plot


# Erzeuge das Animationsobjekt und starte die Animation.
# Wir zeigen dabei nur jeden 4-ten Schritt an.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=range(0, t.size, 4),
                                  blit=True)
plt.show()

