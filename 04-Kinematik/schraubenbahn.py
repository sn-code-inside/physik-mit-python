"""Animation zur Beschleunigung und Bahngeschschwindigkeit bei
einer Bewegung entlang einer Schraubenbahn. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import mpl_toolkits.mplot3d

# Parameter der Simulation.
R = 3.0                  # Radius der Schraubenbahn [m].
T = 8.0                  # Umlaufdauer [s].
dt = 0.05                # Zeitschrittweite [s].
vz = 0.5                 # Geschwindigkeit in z-Richtung [m/s].
N = 5                    # Anzahl der Umläufe.
omega = 2 * np.pi / T    # Winkelgeschwindigkeit [1/s].

# Erzeuge ein Array von Zeitpunkten für N Umläufe.
t = np.arange(0, N * T, dt)

# Erzeuge ein leeres n x 3 - Arrray für die Ortsvektoren.
r = np.empty((t.size, 3))

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
r[:, 0] = R * np.cos(omega * t)
r[:, 1] = R * np.sin(omega * t)
r[:, 2] = vz * t

# Berechne den Geschwindigkeits- und Beschleunigungsvektor.
v = (r[1:, :] - r[:-1, :]) / dt
a = (v[1:, :] - v[:-1, :]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d',
                     elev=40, azim=45)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.grid()


# Definiere einen 3D-Pfeil.
class Arrow3D(mpl.patches.FancyArrowPatch):
    
    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__(posA[0:2], posB[0:2], *args, **kwargs)
        self._pos = np.array([posA, posB])

    def set_positions(self, posA, posB):
        self._pos = np.array([posA, posB])

    def do_3d_projection(self, renderer=None):
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(*self._pos.T, self.axes.M)
        p = np.array(p)
        super().set_positions(p[:, 0], p[:, 1])
        return np.min(p[2, :])


# Plotte die Bahnkurve.
bahn, = ax.plot(r[:, 0], r[:, 1], r[:, 2], linewidth=0.7)

# Erzeuge einen Punkt, der die Position der Masse darstellt.
punkt, = ax.plot([], [], [], 'o', color='red')

# Erzeuge Pfeile für die Geschwindigkeit und die
# Beschleunigung und füge diese der Axes hinzu.
style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
arrow_v = Arrow3D((0, 0, 0), (0, 0, 0),
                  color='red', arrowstyle=style)
arrow_a = Arrow3D((0, 0, 0), (0, 0, 0),
                  color='black', arrowstyle=style)
ax.add_patch(arrow_v)
ax.add_patch(arrow_a)


def update(n):
    # Aktualisiere den Geschwindigkeitspfeil.
    if n < v.shape[0]:
        arrow_v.set_positions(r[n], r[n] + v[n])

    # Aktualisiere den Beschleunigungspfeil.
    if n < a.shape[0]:
        arrow_a.set_positions(r[n], r[n] + a[n])

    # Aktualisiere die Position des Punktes.
    punkt.set_data(np.array([r[n, 0]]), np.array([r[n, 1]]))
    punkt.set_3d_properties(np.array([r[n, 2]]))
  
    return punkt, arrow_v, arrow_a


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30)
plt.show()
