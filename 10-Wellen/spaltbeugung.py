"""Beugungsfeld eines Spaltes durch Überlagerung von
kreisförmigen Wellen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors

# Wellenlänge [m].
lam = 1.0

# Anzahl der Elementarwellen.
N = 100

# Spaltbreite [m].
B = 10

# Lege die Startpositionen jeder Elementarwelle fest.
position = np.zeros((N, 2))
position[:, 0] = np.linspace(-B/2, B/2, N)

# Lege die Phase jeder Elementarwelle fest [rad].
phase = np.zeros(N)

# Lege die Amplitude jeder Elementarwelle fest.
amplitude = B / N * np.ones(N)

# Berechne die die Wellenzahl.
k = 2 * np.pi / lam

# Lege das Gitter für die Auswertung der Wellenfunktion fest.
x1 = np.linspace(-60, 60, 500)
y1 = np.linspace(1, 120, 500)
x, y = np.meshgrid(x1, y1)

# Wir legen nun ein Array komplexer Zahlen der passenden Größe
# an, das mit Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere die komplexe Amplitude jeder Elementarwelle.
for A, (x0, y0), phi0 in zip(amplitude, position, phase):
    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    cos_theta = (y - y0) / d
    u += A * np.exp(1j * (k * d + phi0)) / np.sqrt(d) * cos_theta

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x / $\\lambda$')
ax.set_ylabel('y / $\\lambda$')

# Stelle das Betragsquadrat der Amplitude als Bild dar.
image = ax.imshow(np.abs(u) ** 2,
                  interpolation='bicubic', origin='lower',
                  extent=(np.min(x1), np.max(x1),
                          np.min(y1), np.max(y1)),
                  norm=mpl.colors.LogNorm(vmin=0.001, vmax=2),
                  cmap='inferno')

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Intensität [a.u.]')

# Zeige das Bild an.
plt.show()
