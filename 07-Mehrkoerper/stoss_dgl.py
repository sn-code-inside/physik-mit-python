"""Simulation des schrägen Stoßes zweier kugelförmiger Objekte.

In diesem Programm wird eine elastische Kraft zwischen den
beiden Körpern angenommen, die anfängt zu wirken, sobald sich
die Körper berühren. Die Bewegung wird über die newtonsche
Bewegungsgleichung mit solve_ivp gelöst. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationszeitdauer T und Schrittweite dt [s].
T = 8
dt = 0.02

# Federkonstante beim Aufprall [N/m].
D = 5e3

# Massen der beiden Teilchen [kg].
m1 = 1.0
m2 = 2.0

# Radien der beiden Teilchen [m].
R1 = 0.1
R2 = 0.3

# Anfangspositionen [m].
r0_1 = np.array([-2.0, 0.1])
r0_2 = np.array([0.0, 0.0])

# Anfangsgeschwindigkeiten [m/s].
v0_1 = np.array([1.0, 0.0])
v0_2 = np.array([0.0, 0.0])


def dgl(t, u):
    r1, r2, v1, v2 = np.split(u, 4)

    # Berechne den Abstand der Mittelpunkte.
    dr = np.linalg.norm(r1-r2)

    # Berechne, wie weit die Kugeln ineinander eingedrungen sind.
    dist = max(R1 + R2 - dr, 0)

    # Die Kraft wirkt, sobald sich die Oberflächen berühren.
    F = D * dist

    # Berechne die Vektoren der Beschleunigung. Der
    # Beschleunigungsvektor ist jeweils parallel zur
    # Verbindungslinie der beiden Kugelmittelpunkte.
    er = (r1 - r2) / dr
    a1 = F / m1 * er
    a2 = -F / m2 * er

    # Gib die Zeitableitung des Zustandsvektors zurück.
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Löse die Bewegungsgleichung bis zum Zeitpunkt T.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   max_step=dt,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Berechne Energie und Gesamtimpuls vor und nach dem Stoß und
# gib diese Werte aus.
E0 = 1/2 * (m1 * np.sum(v1[:, 0] ** 2) +
            m2 * np.sum(v2[:, 0] ** 2))
E1 = 1/2 * (m1 * np.sum(v1[:, -1] ** 2) +
            m2 * np.sum(v2[:, -1] ** 2))
p0 = m1 * v1[:, 0] + m2 * v2[:, 0]
p1 = m1 * v1[:, -1] + m2 * v2[:, -1]

print(f'                       vorher          nachher')
print(f'Energie [J]:          {E0:8.5f}      {E1:8.5f}')
print(f'Impuls x [kg m / s]:  {p0[0]:8.5f}      {p1[0]:8.5f}')
print(f'Impuls y [kg m / s]:  {p0[1]:8.5f}      {p1[1]:8.5f}')

# Erstelle eine Figure und eine Axes mit Beschriftung.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-2.0, 2.0])
ax.set_ylim([-1.5, 1.5])
ax.set_aspect('equal')
ax.grid()

# Lege die Linienplots für die Bahnkurve an.
bahn1, = ax.plot([0], [0], '-r', zorder=4)
bahn2, = ax.plot([0], [0], '-b', zorder=3)

# Erzeuge zwei Kreise für die Darstellung der Körper.
kreis1 = mpl.patches.Circle([0, 0], R1, color='red', zorder=4)
kreis2 = mpl.patches.Circle([0, 0], R2, color='blue', zorder=3)
ax.add_artist(kreis1)
ax.add_artist(kreis2)


def update(n):
    # Aktualisiere die Position der beiden Körper.
    kreis1.set_center(r1[:, n])
    kreis2.set_center(r2[:, n])
    # Plotte die Bahnkurve bis zum aktuellen Zeitpunkt.
    bahn1.set_data(r1[0, :n], r1[1, :n])
    bahn2.set_data(r2[0, :n], r2[1, :n])
    return kreis1, kreis2, bahn1, bahn2


# Erstelle die Animaiton und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
