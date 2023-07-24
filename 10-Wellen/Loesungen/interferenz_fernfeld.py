"""Interferenz zweier kreisförmiger Wellen: Vergleich mit der
Fernfeldformel. """

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

# Lege die Amplitude [a.u.] jeder Quelle in einem Array ab.
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

# Erzeuge eine Figure und eine Axes und plotte die Daten.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.grid()
ax.set_xlabel('$\\alpha$ [°]')
ax.set_ylabel('$I / I_0$')
ax.plot(np.degrees(alpha), I_sim, label='Simulation')
ax.plot(np.degrees(alpha), I_theo, label='Fernfeldnäherung')
ax.legend()

# Zeige das Bild an.
plt.show()
