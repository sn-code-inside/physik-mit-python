"""Bahnkurve des schiefen Wurfs ohne Reibung. """

import math
import numpy as np
import matplotlib.pyplot as plt

h = 10.0                       # Anfangshöhe [m].
v_ab = 5.0                     # Abwurfgeschwindigkeit [m/s].
alpha_deg = 25.0               # Abwurfwinkel [°].
g = 9.81                       # Schwerebeschleunigung [m/s²].

# Rechne den Winkel alpha in das Bogenmaß um.
alpha = math.radians(alpha_deg)

# Stelle die Vektoren als 1-dimensionale Arrays dar.
r0 = np.array([0, h])
v0 = np.array([v_ab * math.cos(alpha), v_ab * math.sin(alpha)])
a = np.array([0, -g])

# Berechne den Auftreffzeitpunkt auf dem Boden.
t_e = v0[1] / g + math.sqrt((v0[1] / g)**2 + 2 * r0[1] / g)

# Erzeuge ein 1000 x 1 - Array mit Zeitpunkten.
t = np.linspace(0, t_e, 1000)
t = t.reshape(-1, 1)

# Berechne den Ortsvektor für alle Zeitpunkte im Array t.
r = r0 + v0 * t + 0.5 * a * t**2

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.grid()

# Plotte die Bahnkurve.
ax.plot(r[:, 0], r[:, 1])

# Zeige die Grafik an.
plt.show()
