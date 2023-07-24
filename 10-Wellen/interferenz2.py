"""Interferenz zweier kreisförmiger Wellen (Intensität). """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

# Wellenlänge der ausgestrahlen Wellen [m].
lam = 1.0

# Lege die Amplitude [a.u.] jeder Quelle in einem Array ab.
amplitude = np.array([1.0, 1.0])

# Lege die Phase jeder Quelle in einem Array ab [rad].
phase = np.radians(np.array([0, 0]))

# Lege die Position jeder Quelle in einem N x 2 - Array ab [m].
position = np.array([[-3.0, 0], [3.0, 0]])

# Berechne die Wellenzahl.
k = 2 * np.pi / lam

# Wir wollen die Wellenfunktionen auf einem 500 x 500 - Raster
# von x = -10 m ... +10m und y = -10m .. +10m auswerten.
x1 = np.linspace(-10, 10, 500)
y1 = np.linspace(-10, 10, 500)
x, y = np.meshgrid(x1, y1)

# Wir legen nun ein Array komplexer Zahlen der passenden Größe
# an, das mit Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere für jede Quelle die entsprechende komplexe Amplitude.
for A, (x0, y0), phi0 in zip(amplitude, position, phase):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    u += A * np.exp(1j * (k * r + phi0)) / np.sqrt(r)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

# Stelle das Betragsquadrat der Amplitude als Bild dar.
image = ax.imshow(np.abs(u) ** 2, origin='lower',
                  extent=(np.min(x1), np.max(x1),
                          np.min(y1), np.max(y1)),
                  norm=mpl.colors.LogNorm(vmin=0.01, vmax=10),
                  cmap='inferno',
                  interpolation='bicubic')

# Füge eine Farbbalken hinzu.
fig.colorbar(image, label='Intensität [a.u.]')

# Zeige das Bild an.
plt.show()
