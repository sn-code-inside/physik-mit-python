"""Interferenz zweier kreisförmiger Wellen: Vergleich mit der
Fernfeldformel und Darstellung als Polarplot. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

# Wellenlänge der ausgestrahlen Wellen [m].
lam = 1.0

# Abstand der beiden Quellen voneinander [m].
d = 6.0 * lam

# Abstand, in dem die Simulation ausgewertet wird [m].
R = 10 * d

# Amplitude beider Quellen [a.u.].
A = 1.0

# Lege die Amplitude [a.u.] jeder Quellen in einem Array ab.
amplitude = np.array([A, A])

# Lege die Phase jeder Quelle in einem Array ab [rad].
phase = np.radians(np.array([0, 0]))

# Lege die Position jeder Quelle in einem N x 2 - Array ab [m].
position = np.array([[-d / 2, 0], [d / 2, 0]])

# Berechne die Wellenzahl.
k = 2 * np.pi / lam

# Wir wollen die Wellenfunktionen nur auf einem Kreis mit Radius
# R auswerten.
alpha = np.radians(np.linspace(0, 360, 1000))
x = R * np.sin(alpha)
y = R * np.cos(alpha)

# Wir legen nun ein Array komplexer Zahlen der passenden Größe
# an, das mit Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere für jede Quelle die entsprechende komplexe Amplitude.
for A, (x0, y0), phi0 in zip(amplitude, position, phase):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    u += A * np.exp(1j * (k * r + phi0)) / np.sqrt(r)

# Berechne die Intensität.
I_sim = np.abs(u)**2

# Fernfeldnäherung gemäß der angegebenen Gleichung.
I0 = 1.0 / R
I_theo = 4 * I0 * np.cos(np.pi * d * np.sin(alpha) / lam) ** 2

# Erzeuge eine Figure.
fig = plt.figure()

# Erzeuge eine Axes mit Polarkoordinaten.
ax = fig.add_subplot(1, 1, 1, projection='polar')
ax.grid(True)

# Wir wollen die 0°-Achse nach
# oben zeigen lassen, damit es mit der grafischen Darstellung
# im Buch übereinstimmt.
ax.set_theta_offset(np.pi / 2)

# Die Beschriftung für die r-Achse soll im 80°-Winkel liegen.
ax.set_rlabel_position(80)

# Plotte die beiden Kurven und zeige die Legende an.
ax.plot(alpha, I_sim, label='Simulation')
ax.plot(alpha, I_theo, label='Fernfeldnäherung')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.05))

# Zeige das Bild an.
fig.set_tight_layout(True)
plt.show()
