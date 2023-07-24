"""Darstellung der analytischen Lösung für den schiefen Wurf
ohne Reibung für unterschiedliche Abwurfwinkel. """

import numpy as np
import math
import matplotlib.pyplot as plt

# Anfangshöhe [m].
h = 10.0

# Anfangsgeschwindigkeit [m/s].
v_abwurf = 5.0

# Schwerebeschleunigung [m/s²].
g = 9.81


def wurf(alpha):
    """Gibt ein 500 x 2 - Array zurück, das die Bahnkurve
    bei einem Abwurf unter dem Winkel alpha darstellt. """
    r0 = np.array([0, h])
    v0 = np.array([v_abwurf * math.cos(alpha),
                   v_abwurf * math.sin(alpha)])
    a = np.array([0, -g])

    # Berechne den Zeitpunkt, zu dem der Gegenstand den Boden
    # erreicht.
    t_e = v0[1] / g + math.sqrt((v0[1] / g)**2 + 2 * r0[1] / g)
    
    # Erstelle ein 500 x 1 - Array mit Zeitpunkten.
    t = np.linspace(0, t_e, 500)
    t = t.reshape(-1, 1)

    # Berechne den Ortsvektor für alle Zeitpunkte im Array t.
    # Das Ergebnis ist ein Array der Größe 500 x 2.
    r = r0 + v0 * t + 0.5 * a * t**2

    return r


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid()

# Plotte die Bahnkurve für verschiedene Abwurfwinkel.
for winkel in range(0, 70, 10):
    r = wurf(math.radians(winkel))
    ax.plot(r[:, 0], r[:, 1], label=f'$\\alpha$ = {winkel}°')

# Erzeuge die Legende und zeige die Grafik an.
ax.legend()
plt.show()
