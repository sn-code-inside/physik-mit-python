"""Brechungsgesetz mit ebenen Wellen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Zeitschrittweite [s].
dt = 0.02

# Frequenz der Sender [Hz].
f = 0.25

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c1 = 5.0
c2 = 1.5

# Einfallswinkel [rad].
alpha = np.radians(50)

# Austrittswinkel der gebrochenen Strahlen nach Snellius [rad].
beta = np.arcsin(np.sin(alpha) * c2 / c1)

# Berechne die Kreisfrequenz und die Wellenzahlvektoren.
omega = 2 * np.pi * f
k1 = omega / c1 * np.array([np.sin(alpha), -np.cos(alpha)])
k2 = omega / c2 * np.array([np.sin(beta), -np.cos(beta)])

# Wir wollen die Wellenfunktionen auf einem 200 x 200 - Raster
# von x = -15 m ... +15m und y = -15m .. +15m auswerten.
x1 = np.linspace(-15, 15, 500)
y1 = np.linspace(-15, 15, 500)
x, y = np.meshgrid(x1, y1)

# Erzeuge ein 200 x 200 x 2 - Array, das für jeden Punkt den
# Ortsvektor beinhaltet.
r = np.stack((x, y), axis=2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')

# Stelle eine Array als Bild dar.
image = ax.imshow(0 * x, origin='lower',
                  extent=(np.min(x1), np.max(x1),
                          np.min(y1), np.max(y1)),
                  cmap='jet', clim=(-2, 2),
                  interpolation='bicubic')

# Zeichne eine dünne schwarze Linie, die die Grenze der beiden
# Gebiete darstellt.
linie, = ax.plot([np.min(x1), np.max(x1)], [0, 0], '-k',
                 linewidth=0.5, zorder=5)

# Füge einen Farbbalken hinzu.
fig.colorbar(image, label='Auslenkung [a.u.]')


def update(n):
    # Bestimme die aktuelle Zeit.
    t = dt * n

    # Werte die beiden Wellenfunktionen aus.
    u1 = np.cos(r @ k1 - omega * t)
    u2 = np.cos(r @ k2 - omega * t)

    # Erzeuge ein Array, das in der oberen Halbebene die Welle
    # u1 darstellt und in der unteren Halbebene die Welle u2.
    u = u1
    u[y < 0] = u2[y < 0]

    # Aktualisiere die Bilddaten.
    image.set_data(u)
    return image, linie


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
