"""Simulation eines Masse-Feder-Pendels mit Anregung. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import schraubenfeder

# Simulationszeit T und Zeitschrittweite dt [s].
T = 20.0
dt = 0.01

# Masse des Körpers [kg].
m = 0.1

# Federkonstante [N/m].
D = 2.5

# Reibungskoeffizient [kg / s].
b = 0.05

# Anfangslänge der Feder [m].
L0 = 0.3

# Erdbeschleunigung [m/s²].
g = 9.81

# Anregungskreisfrequenz [1/s].
omega = 6.0

# Anregungsamplitude [m].
A = 0.1

# Anfangsposition der Masse = Gleichgewichtsposition [m].
y0 = -L0 - m * g / D

# Anfangsgeschwindigkeit [m/s].
v0 = 0.0


def y_a(t):
    """Auslenkung des Aufhängepunktes als Funktion der Zeit. """
    return A * np.sin(omega * t)


def dgl(t, u):
    y, v = np.split(u, 2)
    F = D * (y_a(t) - L0 - y) - m * g - b * v
    return np.concatenate([v, F / m])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([y0, v0])

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
y, v = result.y

# Erzeuge eine Figure und ein GridSpec-Objekt.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)
gs = fig.add_gridspec(1, 2, width_ratios=[1, 5])

# Erzeuge zwei Axes-Objekte für die animierte Darstellung und
# den Plot.
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_ylabel('y [m]')
ax1.set_aspect('equal')
ax1.set_xticks([])

ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax2.grid()
ax2.set_xlabel('t [s]')
ax2.tick_params(labelleft=False)
ax2.set_xlim(0, T)

# Erzeuge die Grafikelemente für die Aufhängung, die Feder,
# und die Masse sowie zwei Linienplot für die Auslenkung der
# Masse und der Aufhängung als Funktion der Zeit.
aufhaengung, = ax1.plot([0], [0], 'bo', zorder=5)
feder, = ax1.plot([0], [0], 'k-', zorder=4)
masse, = ax1.plot([0], [0], 'ro', zorder=5)
linie_masse, = ax2.plot(t, y, 'r-')
linie_aufhg, = ax2.plot(t, y_a(t), '-b')


def update(n):
    # Aktualisiere die beiden Linienplots.
    linie_masse.set_data(t[:n + 1], y[:n + 1])
    linie_aufhg.set_data(t[:n + 1], y_a(t[:n + 1]))

    # Aktualisiere die Position der Aufhängung und der Masse.
    aufhaengung.set_data([0], [y_a(t[n])])
    masse.set_data([0], [y[n]])

    # Aktualisiere die Darstellung der Schraubenfeder.
    r0 = np.array([0, y_a(t[n])])
    r1 = np.array([0, y[n]])
    dat = schraubenfeder.data(r0, r1, N=10, R0=0.03,
                              a=0.02, L0=L0)
    feder.set_data(dat)
    return masse, linie_masse, feder, aufhaengung, linie_aufhg


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
