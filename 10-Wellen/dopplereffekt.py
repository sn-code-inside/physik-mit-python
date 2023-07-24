"""Animation zur Entstehung des Doppler-Effekts und des
machschen Kegels. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.005

# Dargestellter Ortsbereich 0..+xmax, 0..+ymax [m].
xlim = (-2.0, 2.0)
ylim = (-1.0, 1.0)

# Frequenz, mit der die Wellenzüge aussendet werden [Hz].
f_Q = 5.0

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 1.0

# Lege die Startposition der Quelle und des Beobachters fest [m].
r0_Q = np.array([-2.0, 0.5])
r0_B = np.array([2.0, -0.5])

# Lege die Geschwindigkeit von Quelle und Beobachter fest [m/s].
v_Q = np.array([0.5, 0])
v_B = np.array([-0.5, 0])

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')

# Erzeuge zwei Kreise für Quelle und Beobachter.
quelle = mpl.patches.Circle((0, 0), radius=0.03,
                            color='black', fill=True, zorder=4)
beobach = mpl.patches.Circle((0, 0), radius=0.03,
                             color='blue', fill=True, zorder=4)
ax.add_patch(quelle)
ax.add_patch(beobach)

# Beschriftung.
ax.text(0.05, 0.95, f'$v_Q = {np.linalg.norm(v_Q) / c:.2f} c$',
        transform=ax.transAxes, color='black', size=12,
        horizontalalignment='left', verticalalignment='top')
ax.text(0.95, 0.95, f'$v_B = {np.linalg.norm(v_B) / c:.2f} c$',
        transform=ax.transAxes, color='blue', size=12,
        horizontalalignment='right', verticalalignment='top')

# Lege eine Liste an, die die kreisförmigen Wellenzüge speichert.
kreise = []


def update(n):
    # Berechne den aktuellen Zeitpunkt.
    t = dt * n

    # Berechne die aktuelle Position von Quelle und Beobachter.
    quelle.center= r0_Q + v_Q * t
    beobach.center = r0_B + v_B * t

    # Erzeuge zum Startzeitpunkt einen neuen Kreis oder wenn
    # seit dem Aussenden des letzten Wellenzuges mehr als eine
    # Periodendauer vergangen ist.
    if not kreise or t >= kreise[-1].startzeit + 1 / f_Q:
        kreis = mpl.patches.Circle(quelle.center, radius=0,
                                   color='red', linewidth=1.5,
                                   fill=False, zorder=3)
        kreis.startzeit = t
        kreise.append(kreis)
        ax.add_patch(kreis)

    # Aktualisiere die Radien aller dargestellen Kreise.
    for kreis in kreise:
        kreis.radius = (t - kreis.startzeit) * c

    # Färbe den Beobachter rot, wenn ein Wellenzug auftrifft.
    beobach.set_color('blue')
    for kreis in kreise:
        d = np.linalg.norm(kreis.center - beobach.center)
        if abs(d - kreis.radius) < beobach.radius:
            beobach.set_color('red')

    # Färbe den Quelle rot, wenn ein Wellezug augesendet wird.
    d = np.linalg.norm(kreise[-1].center - quelle.center)
    if abs(d - kreise[-1].radius) < quelle.radius:
        quelle.set_color('red')
    else:
        quelle.set_color('black')

    return kreise + [quelle, beobach]


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
