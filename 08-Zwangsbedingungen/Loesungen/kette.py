"""Simulation der Bewegung einer Kette aus 5 Punktmassen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Anzahl der Raumdimensionen.
dim = 2

# Simulationsdauer T und Zeitschrittweite dt [s].
T = 20
dt = 0.01

# Lege die Anfangspositionen der Punkte fest [m].
punkte = np.array([[0, 0], [0, -0.5], [0.5, -0.5], [1, -0.5],
                   [1.5, -0.5], [2, -0.5], [2, 0]])

# Massen der einzelnen Körper [kg].
# Der Wert der Masse für die Stützpunkte ist irrelevant.
massen = np.array([0, 1.0, 1.0, 1.0, 1.0, 1.0, 0])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
idx_stuetz = [0, 6]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                   [4, 5], [5, 6]])

# Anzahl der Punkte insgesamt.
n_punkte = punkte.shape[0]

# Anzahl der beweglichen Massen.
N = n_punkte - len(idx_stuetz)

# Anzahl der Zwangsbedingungen.
R = staebe.shape[0]

# Berechne die Länge der Stäbe aus den Anfangspositionen.
laenge = np.empty(R)
for i, stab in enumerate(staebe):
    r1, r2 = punkte[stab]
    laenge[i] = np.linalg.norm(r1 - r2)

# Erzeuge eine Liste mit den Indizes der beweglichen Massen.
idx_knoten = list(set(range(n_punkte)) - set(idx_stuetz))

# Array mit den Komponenten der Anfangspositionen der beweglichen
# Massen [m].
r0 = punkte[idx_knoten].reshape(N * dim)

# Array mit den Komponenten der Anfangsgeschwindigkeit [m/s].
v0 = np.zeros(N * dim)

# Array der Massen für jede Koordinate [kg].
m = np.repeat(massen[idx_knoten], dim)

# Betrag der Erdbeschleunigung [m/s²].
g = 9.81

# Die externe Kraft ist die Schwerkraft in -y-Richtung.
F_ext = np.zeros((N, dim))
F_ext[:, 1] = -g
F_ext = m * F_ext.reshape(-1)

# Parameter für die Baumgarte-Stabilisierung [1/s].
alpha = 10.0
beta = alpha


def h(r):
    """Zwangsbedingungen h_a(r). """
    res = np.zeros(R)

    # Erzeuge ein Array mit den Positionen aller Punkte, wobei
    # die Positionen der beweglichen Massen aus dem Array r
    # übernommen werden.
    r1 = punkte.copy()
    r1[idx_knoten] = r.reshape(N, dim)

    # Die Zwangsbedingungen legen fest, dass die Länge jedes
    # Stabes konstant ist.
    for i, stab in enumerate(staebe):
        ra, rb = r1[stab]
        d = ra - rb
        res[i] = d @ d - laenge[i] ** 2
    return res


def grad_h(r):
    """Gradient g der Zwangsbedingungen.
        g[a, i] =  dh_a / dx_i """
    g = np.zeros((R, N, dim))

    # Erzeuge ein Array mit den Positionen aller Punkte, wobei
    # die Positionen der beweglichen Massen aus dem Array r
    # übernommen werden.
    r1 = punkte.copy()
    r1[idx_knoten] = r.reshape(N, dim)

    # In der Zwangsbedingung taucht der quadratische Abstand
    # der durch einen Stab verbundenen Punkte auf. In der
    # Ableitung steht daher jeweils die doppelte Differenz der
    # Ortsvektoren.
    for i, stab in enumerate(staebe):
        if stab[0] in idx_knoten:
            k = idx_knoten.index(stab[0])
            g[i, k] += 2 * (r1[stab[0]] - r1[stab[1]])
        if stab[1] in idx_knoten:
            k = idx_knoten.index(stab[1])
            g[i, k] += 2 * (r1[stab[1]] - r1[stab[0]])

    return g.reshape(R, N * dim)


def hesse_h(r):
    """Hesse-Matrix H der Zwangsbedingungen.
        H[a, i, j] =  d²h_a / (dx_i dx_j) """
    h = np.zeros((R, N, dim, N, dim))

    # Erstelle eine dim x dim - Einheitsmatrix.
    E = np.eye(dim)

    # Für die Hesse-Matrix muss wieder an den entsprechenden
    # Stellen eine 2 bzw. -2 eingetragen werden.
    for i, stab in enumerate(staebe):
        if stab[0] in idx_knoten:
            k1 = idx_knoten.index(stab[0])
            h[i, k1, :, k1, :] = 2 * E
        if stab[1] in idx_knoten:
            k2 = idx_knoten.index(stab[1])
            h[i, k2, :, k2, :] = 2 * E
        if (stab[0] in idx_knoten) and (stab[1] in idx_knoten):
            h[i, k1, :, k2, :] = -2 * E
            h[i, k2, :, k1, :] = -2 * E

    return h.reshape(R, N * dim, N * dim)


def dgl(t, u):
    r, v = np.split(u, 2)

    # Stelle die Gleichungen für die lambdas auf.
    grad = grad_h(r)
    hesse = hesse_h(r)
    F = - v @ hesse @ v - grad @ (F_ext / m)
    F += - 2 * alpha * grad @ v - beta ** 2 * h(r)
    G = (grad / m) @ grad.T

    # Berechne die lambdas.
    lam = np.linalg.solve(G, F)

    # Berechne die Beschleunigung mithilfe der newtonschen
    # Bewegungsgleichung inkl. Zwangskräften.
    a = (F_ext + lam @ grad) / m

    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-6,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-0.5, 2.5])
ax.set_ylim([-1.5, 0.5])
ax.set_aspect('equal')
ax.grid()

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
plt_stuetz, = ax.plot(punkte[idx_stuetz, 0],
                      punkte[idx_stuetz, 1], 'ro', zorder=5)
plt_knoten, = ax.plot(punkte[idx_knoten, 0],
                      punkte[idx_knoten, 1], 'bo', zorder=5)

# Plotte die Stäbe.
plt_stab = []
for i, stab in enumerate(staebe):
    s, = ax.plot(punkte[stab, 0], punkte[stab, 1], color='black',
                 zorder=4)
    plt_stab.append(s)


def update(n):
    # Erstelle ein Array mit der aktuellen Position aller Punkte.
    r1 = punkte.copy()
    r1[idx_knoten] = r[:, n].reshape(N, dim)

    # Aktualisiere die Position der Massen.
    plt_knoten.set_data(r1[idx_knoten, 0], r1[idx_knoten, 1])

    # Aktualisiere die Stäbe.
    for n, stab in enumerate(staebe):
        plt_stab[n].set_data(r1[stab, 0], r1[stab, 1])

    return plt_stab + [plt_knoten]


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
