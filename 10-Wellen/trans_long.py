"""Animation einer Transversal- und einer Longitudinalwelle. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.005

# Anzahl der dargestellten Massen.
N = 51

# Lege die Ruhepositionen der einzelnen Massenpunkte fest [m].
x = np.linspace(0, 20, N)

# Amplitude A [m] und Frequenz f [Hz] der Welle.
A = 0.8
f = 1.0

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 10.0

# Berechne Kreisfrequenz und Kreiswellenzahl.
omega = 2 * np.pi * f
k = omega / c

# Wähle einen Punkt aus, der rot hervorgehoben dargestellt wird.
ind_rot = N // 2

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 4))

# Erzeuge eine Axes für die Transversalwelle.
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlim(0, np.max(x))
ax1.set_ylim(-1.5 * A, 1.5 * A)
ax1.tick_params(labelbottom=False)
ax1.set_ylabel('y [m]')
ax1.grid()

# Erzeuge eine Axes für die Longitudinalwelle.
ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
ax2.set_ylim(-1.5 * A, 1.5 * A)
ax2.set_xlabel('x [m]')
ax2.set_ylabel('y [m]')
ax2.grid()

# Erzeuge je einen Punktplot (blau) für die beiden Wellen.
trans, = ax1.plot(x, 0*x, 'ob', zorder=6)
long, = ax2.plot(x, 0*x, 'ob', zorder=6)

# Erzeuge je einen Punktplot (rot) für je einen Massenpunkt.
trans1, = ax1.plot([x[ind_rot]], [0], 'or', zorder=7)
long1, = ax2.plot([x[ind_rot]], [0], 'or', zorder=7)

# Erzeuge  je einen Plot mit kleinen Punkten für die Darstellung
# der Ruhelage.
ax1.plot(x, 0*x, '.', color='lightblue', zorder=4)
ax2.plot(x, 0*x, '.', color='lightblue', zorder=4)
ax1.plot([x[ind_rot]], [0], '.', color='pink', zorder=5)
ax2.plot([x[ind_rot]], [0], '.', color='pink', zorder=5)


def update(n):
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Berechne die Momentanauslenkung u der Welle.
    u = A * np.cos(k * x - omega * t)

    # Aktualisiere die beiden Darstellungen.
    trans.set_ydata(u)
    long.set_xdata(x + u)

    # Aktualisiere die Position des herovrgehobenen Punktes.
    trans1.set_ydata(u[ind_rot])
    long1.set_xdata(x[ind_rot] + u[ind_rot])

    return long, trans, long1, trans1


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
