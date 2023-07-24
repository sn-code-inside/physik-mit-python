"""Simulation eines Gases: Berechnung der Kraft auf die Wände. """

import numpy as np
import matplotlib.pyplot as plt

# Anzahl der Raumdimensionen.
dim = 2

# Anzahl der Teilchen.
N = 100

# Simulationszeitdauer T und Schrittweite dt [s].
T = 20
dt = 2

# Die Teilchen erhalten am Anfang alle eine Geschwidigkeit mit
# dem Betrag v_avg und zufälliger Richtung.
v_avg = 2

# Kleinste Zeitdifferenz, bei der Stöße als gleichzeitig
# angenommen werden [s].
epsilon = 1e-9

# Für jede Wand wird der Abstand vom Koordinatenursprung
# wand_d und ein nach außen zeigender Normalenvektor wand_n
# angegeben.
wand_d = np.array([2.0, 2.0, 2.0, 2.0])
wand_n = np.array([[0, -1.0], [0, 1.0], [-1.0, 0], [1.0, 0]])

# Positioniere die Massen zufällig im Bereich
# x= -1,9 ... 1,9 m und y = -1,9 ... 1,9 m.
r0 = 1.9 * (2 * np.random.rand(N, dim) - 1)

# Wähle zufällige Geschwindigkeiten, die auf v_avg normiert sind.
v0 = -0.5 + np.random.rand(N, dim)
v0 *= v_avg / np.linalg.norm(v0, axis=1).reshape(-1, 1)

# Alle Teilchen bekommen den gleichen Radius [m].
radius = 0.05 * np.ones(N)

# Alle Teilchen bekommen die gleiche Masse [kg].
m = np.ones(N)

# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, T, dt)
r = np.empty((t.size, N, dim))
v = np.empty((t.size, N, dim))
F = np.zeros(t.size)
r[0] = r0
v[0] = v0


def koll_teil(r, v):
    """Gibt die Zeit bis zur nächsten Teilchenkollision und
    die Indizes der beteiligten Teilchen zurück. """

    # Erstelle N x N x dim - Arrays, die die paarweisen Orts- und
    # Geschwindigkeitesdifferenzen enthalten:
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
    # gelöst werden. Dabei ist nur die kleinere Lösung relevant.
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
dt_teilchen, teilchen1, teilchen2 = koll_teil(r[0], v[0])
dt_wand, teilchen_w, wand = koll_wand(r[0], v[0])
dt_koll = min(dt_teilchen, dt_wand)

# Schleife über die Zeitschritte.
for i in range(1, t.size):
    # Kopiere die Positionen aus dem vorherigen Zeitschritt.
    r[i] = r[i - 1]
    v[i] = v[i - 1]

    # Zeit, die in diesem Zeitschritt schon simuliert wurde.
    t1 = 0

    # Behandle nacheinander alle Kollisionen in diesem
    # Zeitschritt.
    while t1 + dt_koll <= dt:
        # Bewege alle Teilchen bis zur Kollision vorwärts.
        r[i] += v[i] * dt_koll

        # Führe die Kollision zwischen Teilchen durch.
        if dt_teilchen <= dt_wand:
            for k1, k2 in zip(teilchen1, teilchen2):
                dr = r[i, k1] - r[i, k2]
                dv = v[i, k1] - v[i, k2]
                er = dr / np.linalg.norm(dr)
                m1 = m[k1]
                m2 = m[k2]
                v1_s = v[i, k1] @ er
                v2_s = v[i, k2] @ er
                v1_s_neu = (2 * m2 * v2_s +
                            (m1 - m2) * v1_s) / (m1 + m2)
                v2_s_neu = (2 * m1 * v1_s +
                            (m2 - m1) * v2_s) / (m1 + m2)
                v[i, k1] += -v1_s * er + v1_s_neu * er
                v[i, k2] += -v2_s * er + v2_s_neu * er

        # Führe die Kollision zwischen Teilchen und Wänden durch.
        if dt_teilchen >= dt_wand:
            for n, w in zip(teilchen_w, wand):
                v1_s = v[i, n] @ wand_n[w]
                v[i, n] -= 2 * v1_s * wand_n[w]
                # Addiere den Impulsübertrag auf die Wand.
                F[i] += 2 * m[n] * v1_s

        # Innerhalb dieses Zeitschritts wurde eine Zeitdauer
        # dt_koll bereits behandelt.
        t1 += dt_koll

        # Da Kollisionen stattgefunden haben, müssen wir diese
        # neu berechnen.
        dt_teilchen, teilchen1, teilchen2 = koll_teil(r[i], v[i])
        dt_wand, teilchen_w, wand = koll_wand(r[i], v[i])
        dt_koll = min(dt_teilchen, dt_wand)

    # Bis zum Ende des aktuellen Zeitschrittes (dt) finden nun
    # keine Kollision mehr statt. Wir bewegen alle Teilchen bis
    # zum Ende des Zeitschritts vorwärts und müssen nicht
    # erneut nach Kollisionen suchen.
    r[i] = r[i] + v[i] * (dt - t1)
    dt_koll -= dt - t1

    # Gib eine Information zum Fortschritt der Simulation in
    # Prozent aus.
    print(f'{100*i/t.size:.1f} %')

# Teile die Impulsüberträge pro Wand durch die jeweilige Länge
# des Zeitintervalls. Auf diese Weise erhalten wir die mittlere
# Kraft in jedem einzelnen Intervall.
F /= dt

# Wir ignorieren die ersten beiden Einträge des Kraftarrays.
# Der Eintrag F[0] ist immer Null und im Eintrag F[1] spielen
# die Anfangsbedingungen eventuell noch eine wesentliche Rolle.
F_avg = np.mean(F[2:])
F_err = np.std(F[2:], ddof=1) / np.sqrt(F.size - 2)
print(f'Mittlerer Normalkraft: {F_avg:.1f} +- {F_err:.1f} N')

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('t [s]')
ax.set_ylabel('Normalkraft [N]')
ax.grid()

# Plotte die Kraft als Funktion der Zeit.
ax.plot(t, F, 'o')

# Zeige die Grafik an.
plt.show()
