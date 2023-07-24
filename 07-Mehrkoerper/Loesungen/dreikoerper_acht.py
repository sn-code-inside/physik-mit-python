"""Simulation von drei Körpern auf einer 8-förmigen Bahn. """

import numpy as np
import scipy.integrate
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anzahl der Körper und Raumdimension.
N = 3
dim = 2

# Ein Tag [s] und ein Jahr [s].
tag = 24 * 60 * 60
jahr = 365.25 * tag

# Eine Astronomische Einheit [m].
AE = 1.495978707e11

# Simulationszeitdauer T, Schrittweite dt [s].
T = 20 * jahr
dt = 1 * tag

# Newtonsche Graviationskonstante [m³ / (kg * s²)].
G = 6.674e-11

# Massen der Körper [kg].
m0 = 2e30
m = m0 * np.ones(3)

# Abstand der Körper vom Schwerpunkt.
d = 1 * AE

# Anfangspositionen der Körper [m].
r0 = d * np.array([[1.0, 0.0],
                   [-1.0, 0.0],
                   [0.0, 0.0]])

# Lege die Geschwindigkeit der Masse m3 fest.
alpha = np.radians(56.9)
v3_norm = 1.27 * np.sqrt(G * m0 / d)
v3 = v3_norm * np.array([np.cos(alpha), np.sin(alpha)])

# Anfangsgeschwindigkeit der Körper.
v0 = np.array([-v3/2, -v3/2, v3])

# Farben für die drei Körper.
farbe = ['red', 'green', 'blue']


def dgl(t, u):
    r, v = np.split(u, 2)
    r = r.reshape(N, dim)
    a = np.zeros((N, dim))
    for i in range(N):
        for j in range(i):
            dr = r[j] - r[i]
            gr = G / np.linalg.norm(dr) ** 3 * dr
            a[i] += gr * m[j]
            a[j] -= gr * m[i]
    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionals Array um:
#    1. Index - Himmelskörper
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)

# Berechne die verschiedenen Energiebeiträge.
E_kin = 1/2 * m @ np.sum(v * v, axis=1)
E_pot = np.zeros(t.size)
for i in range(N):
    for j in range(i):
        dr = np.linalg.norm(r[i] - r[j], axis=0)
        E_pot -= G * m[i] * m[j] / dr
E = E_pot + E_kin
dE_rel = (np.max(E) - np.min(E)) / E[0]
print(f'Relative Energieänderung: {dE_rel:.2g}')

# Erzeuge eine Figure und eine Axes für die Animation.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [AE]')
ax.set_ylabel('y [AE]')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-1.2, 1.2])
ax.set_aspect('equal')
ax.grid()

# Plotte für jeden Planeten die Bahnkurve und füge die
# Beschriftungen hinzu.
for i in range(N):
    ax.plot(r[i, 0] / AE, r[i, 1] / AE, '-',
            color=farbe[i], linewidth=0.2)

# Erzeuge für jeden Planeten einen Punktplot in der
# entsprechenden Farbe und speichere diesen in der Liste planet.
planet = []
for i in range(N):
    p, = ax.plot([0], [0], 'o', color=farbe[i])
    planet.append(p)

# Füge ein Textfeld für die Anzeige der verstrichenen Zeit hinzu.
text = ax.text(-1.1, 1.1, '')


def update(n):
    for i in range(N):
        planet[i].set_data(r[i, 0, n] / AE, r[i, 1, n] / AE)
    text.set_text(f'{n * dt / jahr:.2f} Jahre')
    return planet + [text]


# Zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size)
plt.show()
