"""Beschleunigung und Geschwindigkeit bei einem Pendel. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Länge des Pendels [m].
L = 3.0

# Schwingungsdauer [s].
T = 3.47

# Zeitschrittweite [s].
dt = 0.02

# Anfangsauslenkung [rad].
phi0 = np.radians(10)

# Erzeuge ein Array von Zeitpunkten für eine Periode.
t = np.arange(0, T, dt)

# Erzeuge ein leeres n x 2 - Arrray für den Ortsvektor des
# Massenpunktes x-y-Ebene.
r = np.empty((t.size, 2))

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
omega = 2 * np.pi / T
phi = phi0 * np.cos(omega * t)
r[:, 0] = L * np.sin(phi)
r[:, 1] = -L * np.cos(phi)

# Berechne den Geschwindigkeits- und Beschleunigungsvektor
# durch den Differenzenquotient.
v = (r[1:, :] - r[:-1, :]) / dt
a = (v[1:, :] - v[:-1, :]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-0.5 * L, 0.5 * L)
ax.set_ylim(-1.2 * L, 0.1 * L)
ax.set_aspect('equal')
ax.grid()

# Plotte die Kreisbahn und den Faden des Pendels.
bahn, = ax.plot(r[:, 0], r[:, 1])
faden, = ax.plot([0, 0], [0, 0], color='black', lw=0.5)

# Erzeuge einen Kreis, der die Position der Masse darstellt.
Punkt, = ax.plot([0], [0], 'o', color='blue')

# Erzeuge zwei Pfeile für die Geschwindigkeit und die
# Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
arrow_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_artist(arrow_v)
ax.add_artist(arrow_a)


def update(n):
    # Aktualisiere den Geschwindigkeitspfeil.
    if n < v.shape[0]:
        arrow_v.set_positions(r[n], r[n] + v[n])

    # Aktualisiere den Beschleunigungspfeil.
    if n < a.shape[0]:
        arrow_a.set_positions(r[n], r[n] + a[n])

    # Aktualisiere die Darstellung des Fadens.
    faden.set_data([0, r[n, 0]], [0, r[n, 1]])

    # Aktualisiere die Position des Punktes.
    Punkt.set_data(r[n])

    return Punkt, arrow_v, arrow_a, faden


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
