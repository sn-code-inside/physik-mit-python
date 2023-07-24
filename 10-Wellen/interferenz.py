"""Interferenz zweier kreisförmiger Wellen (Animation). """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.02

# Frequenz der Quelle [Hz].
f = 1.0

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 1.0

# Lege die Amplitude [a.u.] jeder Quelle in einem Array ab.
amplitude = np.array([1.0, 1.0])

# Lege die Phase jeder Quelle in einem Array ab [rad].
phase = np.radians(np.array([0, 0]))

# Lege die Position jeder Quelle in einem N x 2 - Array ab [m].
position = np.array([[-3.0, 0], [3.0, 0]])

# Berechne die Kreisfrequenz und die Wellenzahl.
omega = 2 * np.pi * f
k = omega / c

# Wir wollen die Wellenfunktionen auf einem 500 x 500 - Raster
# von x = -10 m ... +10m und y = -10m .. +10m auswerten.
x1 = np.linspace(-10, 10, 500)
y1 = np.linspace(-10, 10, 500)
x, y = np.meshgrid(x1, y1)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

# Stelle ein 2-dimensionales Array als Bild dar.
image = ax.imshow(0 * x, origin='lower',
                  extent=(np.min(x1), np.max(x1),
                          np.min(y1), np.max(y1)),
                  cmap='jet', clim=(-2, 2),
                  interpolation='bicubic')

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Auslenkung [a.u.]')


def update(n):
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Lege ein mit Nullen gefülltes Array der passenden Größe an.
    u = np.zeros(x.shape)

    # Addiere nacheinander die Wellenfelder jeder Quelle.
    for A, (x0, y0), phi0 in zip(amplitude, position, phase):
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        u1 = A * np.sin(omega * t - k * r + phi0) / np.sqrt(r)
        u1[omega * t - k * r < 0] = 0
        u += u1

    # Aktualisiere die Bilddaten.
    image.set_data(u)
    return image,


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=0, blit=True)
plt.show()
