"""Simulation der elastischen Stöße mehrerer Teilchen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anzahl der Raumdimensionen.
dim = 2

# Simulationszeitdauer T und Schrittweite dt [s].
T = 100
dt = 0.005

# Anfangspositionen [m].
r0 = np.array([[-1.0, 0.0],  [0.5, 0.0], [0.45, -0.05],
               [0.45, 0.05], [0.55, -0.05], [0.55, 0.05]])

# Anzahl der Teilchen.
N = r0.shape[0]

# Anfangsgeschwindigkeiten [m/s].
v = np.zeros((N, dim))
v[0] = np.array([3.0, 0.0])

# Radien der einzelnen Teilchen [m].
radius = 0.03 * np.ones(N)

# Massen der Teilchen [kg].
m = 0.2 * np.ones(N)

# Für jede Wand wird der Abstand vom Koordinatenursprung
# wand_d und ein nach außen zeigender Normalenvektor wand_n
# angegeben.
wand_d = np.array([1.2, 1.2, 0.6, 0.6])
wand_n = np.array([[-1.0, 0], [1.0, 0], [0, -1.0], [0, 1.0]])

# Kleinste Zeitdifferenz, bei der Stöße als gleichzeitig
# angenommen werden [s].
epsilon = 1e-9

# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, T, dt)
r = np.empty((t.size, N, dim))
r[0] = r0


def koll_teil(r, v):
    """Gibt die Zeit bis zur nächsten Teilchenkollision und die
    Indizes der beteiligten Teilchen zurück. """

    # Erstelle N x N x dim - Arrays, die die paarweisen
    # Orts- und Geschwindigkeitsdifferenzen enthalten:
    # dr[i, j] ist der Vektor r[i] - r[j]
    # dv[i, j] ist der Vektor v[i] - v[j]
    dr = r.reshape(N, 1, dim) - r
    dv = v.reshape(N, 1, dim) - v

    # Erstelle ein N x N - Array, das das Betragsquadrat der
    # Vektoren aus dem Array dv enthält.
    dv2 = np.sum(dv * dv, axis=2)

    # Erstelle ein N x N - Array, das die paarweise Summe
    # der Radien der Teilchen enthält.
    rad = radius + radius.reshape(N, 1)

    # Um den Zeitpunkt der Kollision zu bestimmen, muss eine
    # quadratische Gleichung der Form
    #          t² + 2 a t + b = 0
    # gelöst werden. Nur die kleinere Lösung ist relevant.
    a = np.sum(dv * dr, axis=2) / dv2
    b = (np.sum(dr * dr, axis=2) - rad ** 2) / dv2
    D = a**2 - b
    t = -a - np.sqrt(D)

    # Suche den kleinsten positiven Zeitpunkt einer Kollision.
    t[t <= 0] = np.NaN
    t_min = np.nanmin(t)

    # Suche die entsprechenden Teilchenindizes heraus.
    i, j = np.where(np.abs(t - t_min) < epsilon)

    # Wähle die erste Hälfte der Indizes aus, da jede
    # Teilchenpaarung doppelt auftritt.
    i = i[0:i.size // 2]
    j = j[0:j.size // 2]

    # Gib den Zeitpunkt und die Teilchenindizes zurück. Wenn
    # keine Kollision stattfindet, dann gib inf zurück.
    if np.isnan(t_min):
        t_min = np.inf

    return t_min, i, j


def koll_wand(r, v):
    """Gibt die Zeit bis zur nächsten Wandkollision, den Index
    der Teilchen und den Index der Wand zurück. """

    # Berechne den Zeitpunkt der Kollision der Teilchen mit
    # einer der Wände. Das Ergebnis ist ein N x Wandanzahl -
    # Array.
    entfernung = wand_d - radius.reshape(-1, 1) - r @ wand_n.T
    t = entfernung / (v @ wand_n.T)

    # Ignoriere alle nichtpositiven Zeiten.
    t[t <= 0] = np.NaN

    # Ignoriere alle Zeitpunkte, bei denen sich das Teilchen
    # entgegen den Normalenvektor bewegt. Eigentlich dürfte
    # so etwas gar nicht vorkommen, aber aufgrund von
    # Rundungsfehlern kann es passieren, dass ein Teilchen
    # sich leicht außerhalb einer Wand befindet.
    t[(v @ wand_n.T) < 0] = np.NaN

    # Suche den kleinsten Zeitpunkt, und gib die Zeit und die
    # Indizes zurück.
    t_min = np.nanmin(t)
    teilchen, wand = np.where(np.abs(t - t_min) < epsilon)
    return t_min, teilchen, wand


# Berechne die Zeitdauer bis zur ersten Kollision und die
# beteiligten Partner.
dt_teilchen, teilchen1, teilchen2 = koll_teil(r[0], v)
dt_wand, teilchen_w, wand = koll_wand(r[0], v)
dt_koll = min(dt_teilchen, dt_wand)

# Schleife über die Zeitschritte.
for i in range(1, t.size):
    # Kopiere die Positionen aus dem vorherigen Zeitschritt.
    r[i] = r[i - 1]

    # Zeit, die in diesem Zeitschritt schon simuliert wurde.
    t1 = 0

    # Behandle nacheinander alle Kollisionen in diesem
    # Zeitschritt.
    while t1 + dt_koll <= dt:
        # Bewege alle Teilchen bis zur Kollision vorwärts.
        r[i] += v * dt_koll

        # Kollisionen zwischen Teilchen untereinander.
        if dt_teilchen <= dt_wand:
            for k1, k2 in zip(teilchen1, teilchen2):
                dr = r[i, k1] - r[i, k2]
                dv = v[k1] - v[k2]
                er = dr / np.linalg.norm(dr)
                m1 = m[k1]
                m2 = m[k2]
                v1_s = v[k1] @ er
                v2_s = v[k2] @ er
                v1_s_neu = (2 * m2 * v2_s +
                            (m1 - m2) * v1_s) / (m1 + m2)
                v2_s_neu = (2 * m1 * v1_s +
                            (m2 - m1) * v2_s) / (m1 + m2)
                v[k1] += (v1_s_neu - v1_s) * er
                v[k2] += (v2_s_neu - v2_s) * er

        # Kollisionen zwischen Teilchen und Wänden.
        if dt_teilchen >= dt_wand:
            for n, w in zip(teilchen_w, wand):
                v1_s = v[n] @ wand_n[w]
                v[n] -= 2 * v1_s * wand_n[w]

        # Innerhalb dieses Zeitschritts wurde damit eine
        # Zeitdauer dt_koll bereits behandelt.
        t1 += dt_koll

        # Da Kollisionen stattgefunden haben, müssen wir diese
        # neu berechnen.
        dt_teilchen, teilchen1, teilchen2 = koll_teil(r[i], v)
        dt_wand, teilchen_w, wand = koll_wand(r[i], v)
        dt_koll = min(dt_teilchen, dt_wand)

    # Bis zum Ende des aktuellen Zeitschrittes (dt) finden nun
    # keine Kollision mehr statt. Wir bewegen alle Teilchen
    # bis zum Ende des Zeitschritts vorwärts und müssen nicht
    # erneut nach Kollisionen suchen.
    r[i] += v * (dt - t1)
    dt_koll -= dt - t1


# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.2, 1.2])
ax.set_ylim([-0.6, 0.6])
ax.set_aspect('equal')
ax.grid()

# Erzeuge für jedes Teilchen einen Kreis.
kreis = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i])
    ax.add_artist(c)
    kreis.append(c)


def update(n):
    for i in range(N):
        kreis[i].set_center(r[n, i])
    return kreis


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
