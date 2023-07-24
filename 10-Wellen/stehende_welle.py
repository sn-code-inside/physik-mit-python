"""Animation einer stehenden Welle (1d). """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.01

# Dargestellter Bereich von x=-x_max bis x=x_max [m].
x_max = 20.0

# Amplitude [a.u.] und Frequenz [Hz] der Welle von links.
A1 = 1.0
f1 = 1.0

# Amplitude [a.u.] und Frequenz [Hz] der Welle von rechts.
A2 = 1.0
f2 = 1.0

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 10.0

# Berechne Kreisfrequenz und Kreiswellenzahl.
omega1 = 2 * np.pi * f1
omega2 = 2 * np.pi * f2
k1 = omega1 / c
k2 = omega2 / c

# Erzeuge eine x-Achse.
x = np.linspace(-x_max, x_max, 500)

# Erzeuge eine Figure und eines Axes.
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-x_max, x_max)
ax.set_ylim(-1.2 * (A1 + A2), 1.2 * (A1 + A2))
ax.set_xlabel('x [m]')
ax.set_ylabel('$u$')
ax.grid()

# Erzeuge die Plots für die beiden Wellen und deren Summe.
welle1, = ax.plot(x, 0 * x, '-r',
                  zorder=5, linewidth=2, label='von links')
welle2, = ax.plot(x, 0 * x, '-b',
                  zorder=5, linewidth=2, label='von rechts')
welle3, = ax.plot(x, 0 * x, '-k',
                  zorder=2, linewidth=2.5, label='Überlagerung')

# Füge die entsprechenden Legendeneinträge hinzu.
ax.legend(loc='upper right', ncol=2)


def update(n):
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Berechne die Momentanauslenkung u1 der ersten Welle.
    phi1 = omega1 * t - k1 * (x + x_max)
    u1 = A1 * np.sin(phi1)
    u1[phi1 < 0] = 0

    # Berechne die Momentanauslenkung u2 der zweiten Welle.
    phi2 = omega2 * t + k2 * (x - x_max)
    u2 = A2 * np.sin(phi2)
    u2[phi2 < 0] = 0

    # Aktualisiere die beiden Darstellungen.
    welle1.set_ydata(u1)
    welle2.set_ydata(u2)

    # Aktualisiere die Überlagerung.
    welle3.set_ydata(u1 + u2)

    return welle1, welle2, welle3


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
