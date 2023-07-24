"""Demonstration einer Resonanzkurve mit Hysterese. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Simulationsdauer [s].
T = 60

# Anfängliche Federhärte [N/m].
D0 = 400.0

# Masse [kg].
m = 1e-3

# Reibungskoeffizient [kg/s].
b = 0.1

# Konstante für den nichtlinearen Term [m].
alpha = 5e-3

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


# Zustandsvektor für den Anfangszustand.
u0 = np.array([0, 0])

# Löse die Differentialgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-4)
t = result.t
x, v = result.y

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Plotte den Zeitverlauf der erzwungenen Schwingung.
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_ylabel('Auslenkung [m]')
ax1.tick_params(labelbottom=False)
ax1.set_xlim(0, T)
ax1.grid()
ax1.plot(t, x)

# Plotte die Frequenz der Anregung.
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('$\\omega$ [1/s]')
ax2.grid()
ax2.plot(t, omega_a(t))

# Zeige den Plot an.
plt.show()
