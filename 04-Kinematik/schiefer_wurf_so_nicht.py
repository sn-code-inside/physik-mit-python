"""Bahnkurve des schiefen Wurfs: Funktioniert so nicht! """

import math
import numpy as np
import matplotlib.pyplot as plt

h = 10.0                       # Anfangshöhe [m].
v_ab = 5.0                     # Abwurfgeschwindigkeit [m/s].
alpha_deg = 25.0               # Abwurfwinkel [°].
g = 9.81                       # Schwerebeschleunigung [m/s²].

# Rechne den Winkel in das Bogenmaß um.
alpha = math.radians(alpha_deg)

# Stelle die Vektoren als 1-dimensionale Arrays dar.
r0 = np.array([0, h])
v0 = np.array([v_ab * math.cos(alpha), v_ab * math.sin(alpha)])
a = np.array([0, -g])

# Berechne den Auftreffzeitpunkt auf dem Boden.
t_e = v0[1] / g + math.sqrt((v0[1] / g) ** 2 + 2 * r0[1] / g)

# Erezuge ein Array von Zeitpunkten.
t = np.linspace(0, t_e, 1000)

# Berechne den Ortsvektor für diese Zeitpunkte.
r = r0 + v0 * t + 0.5 * a * t**2
