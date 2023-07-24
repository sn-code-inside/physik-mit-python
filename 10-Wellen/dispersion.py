"""Effekt der Gruppengeschwindigkeit am Beispiel der
Überlagerung zweier sinusförmiger Wellen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.05

# Dargestellter Bereich von x=0 bis x=x_max [m].
x_max = 30.0

# Wellenlängen der beiden Wellen [m].
lambda1 = 0.95
lambda2 = 1.05

# Phasengeschwindigkeit der beiden Wellen [m/s].
c1 = 0.975
c2 = 1.025

# Berechne die Wellenzahlen und die Kreisfrequenzen.
k1 = 2 * np.pi / lambda1
k2 = 2 * np.pi / lambda2
omega1 = c1 * k1
omega2 = c2 * k2

# Berechne die Gruppengeschwindigkeit.
c_gr = (omega1 - omega2) / (k1 - k2)

# Berechne die mittlere Phasengeschwindigkeit.
c_ph = (c1 + c2) / 2

# Lege ein Array von x-Werten an.
x = np.linspace(0, x_max, 1000)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('Auslenkung [a.u.]')
ax.set_xlim(np.min(x), np.max(x))
ax.set_ylim(-2.2, 2.2)

# Erzeuge drei Linienplots für die Wellen.
welle1, = ax.plot(x, 0 * x, '-r', zorder=5, linewidth=1)
welle2, = ax.plot(x, 0 * x, '-b', zorder=5, linewidth=1)
welle3, = ax.plot(x, 0 * x, '-k', zorder=2, linewidth=2)

# Erzeuge zwei Linienplots zur Darstellung der Geschwindigkeit.
linie1, = ax.plot([0], [0], '-', color='gray',
                  zorder=1, linewidth=4)
linie2, = ax.plot([0], [0], '-m', zorder=1, linewidth=4)


def update(n):
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Werte die Wellenfunktionen aus.
    u1 = np.cos(k1 * x - omega1 * t)
    u2 = np.cos(k2 * x - omega2 * t)

    # Stelle die beiden Wellen und ihre Überlagerung dar.
    welle1.set_ydata(u1)
    welle2.set_ydata(u2)
    welle3.set_ydata(u1 + u2)

    # Bewege die erste Linie mit der Gruppengeschwindigkeit.
    pos1 = (t * c_gr) % x_max
    linie1.set_ydata(ax.get_ylim())
    linie1.set_xdata([pos1, pos1])

    # Bewege die zweite Linie mit der Phasengeschwindigkeit.
    pos2 = (t * c_ph) % x_max
    linie2.set_ydata(ax.get_ylim())
    linie2.set_xdata([pos2, pos2])

    return welle1, welle2, welle3, linie1, linie2


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
