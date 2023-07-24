"""Simulation von Stößen zwischen vielen kugelförmigen
Objekten mithilfe des Differentialgleichungsansatzes. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Raumdimensionen.
dim = 2

# Anzahl der Teilchen.
N = 10

# Simulationszeitdauer T und Schrittweite dt [s].
T = 10
dt = 0.02

# Federkonstante beim Aufprall [N/m].
D = 5e3

# Positioniere die Massen zufällig im Bereich
# x=0,5 ... 1,5 und y = 0,5 ... 1,5 [m].
r0 = 0.5 + np.random.rand(N, dim)

# Wähle zufällige Geschwindigkeiten im Bereich
# vx = -0,5 ... 0,5 und vy = -0,5 ... 0,5 [m/s]
v0 = -0.5 + np.random.rand(N, dim)

# Wähle zufällige Radien im Bereich von 0,02 bis 0,04 [m].
radius = 0.02 + 0.02 * np.random.rand(N)

# Wähle zufällige Massen im Berevon von 0,2 bis 2,0 [kg].
m = 0.2 + 1.8 * np.random.rand(N)


def dgl(t, u):
    r, v = np.split(u, 2)
    r = r.reshape(N, dim)
    a = np.zeros((N, dim))
    for i in range(N):
        for j in range(i):
            # Berechne den Abstand der Mittelpunkte.
            dr = np.linalg.norm(r[i] - r[j])
            # Berechne die Eindringtiefe.
            dist = max(radius[i] + radius[j] - dr, 0)
            # Die Kraft soll proportional zur Eindringtiefe sein.
            F = D * dist
            er = (r[i] - r[j]) / dr
            a[i] += F / m[i] * er
            a[j] -= F / m[j] * er
    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

# Löse die Bewegungsgleichung bis zum Zeitpunkt T.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, max_step=dt,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionals Array um:
#    1. Index - Teilchen
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([0, 2])
ax.set_ylim([0, 2])
ax.set_aspect('equal')
ax.grid()

# Füge die Grafikobjekte zur Axes hinzu.
kugel = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i])
    ax.add_artist(c)
    kugel.append(c)


def update(n):
    for i in range(N):
        kugel[i].set_center(r[i, :, n])
    return kugel


# Erstelle die Animaiton und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
