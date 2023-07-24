"""Energieverteilung in einem Gas mit verschiedenen Teilchen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anzahl der Raumdimensionen.
dim = 2

# Anzahl der Teilchen.
N = 200

# Simulationszeitdauer T und Schrittweite dt [s].
T = 10
dt = 0.01

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

# Wähle zufällige Geschwindigkeiten mit Betrag 1 m/s.
v0 = -0.5 + np.random.rand(N, dim)
v0 /= np.linalg.norm(v0, axis=1).reshape(-1, 1)

# Wähle zufällige Radien zwischen 0.01 m und 0.05 m.
radius = 0.01 + 0.04 * np.random.rand(N)

# Wähle zufällige Massen zwischen 0.01 kg un 1.00 kg.
m = 0.01 + 0.99 * np.random.rand(N)

# Maximale angezeigte Energie im Diagramm [J].
E_max = 1.0

# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, T, dt)
r = np.empty((t.size, N, dim))
v = np.empty((t.size, N, dim))
r[0] = r0
v[0] = v0


def koll_teil(r, v):
    """Gibt die Zeit bis zur nächsten Teilchenkollision und
    die Indizes der beteiligten Teilchen zurück. """

    # Erstelle N x N x dim - Arrays, die die paarweisen
    # Orts- und Geschwindigkeitesdifferenzen enthalten:
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

        # Kollisionen zwischen Teilchen untereinander.
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

        # Kollisionen zwischen Teilchen und Wänden.
        if dt_teilchen >= dt_wand:
            for n, w in zip(teilchen_w, wand):
                v1_s = v[i, n] @ wand_n[w]
                v[i, n] -= 2 * v1_s * wand_n[w]

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


# Berechne die kinetische Energie für jeden Zeitschritt und für
# jedes Teilchen.
E_kin = 1 / 2 * m * np.sum(v ** 2, axis=2)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(8, 4))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Animation der Bewegung der Teilchen.
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title('Teilchenbewegung')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_xlim([-2.1, 2.1])
ax1.set_ylim([-2.1, 2.1])
ax1.set_xlim([-2.1, 2.1])
ax1.grid()

# Erstelle die Farbtabelle und erzeuge ein Objekt, das jeder
# Masse eine Farbe zuordnet.
mapper = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
mapper.set_array(m)
mapper.autoscale()

# Erzeuge für jedes Teilchen einen Kreis mit passendem Radius.
kreis = []
for i in range(N):
    c = mpl.patches.Circle([0, 0], radius[i],
                           color=mapper.to_rgba(m[i]))
    ax1.add_artist(c)
    kreis.append(c)

# Erzeuge eine zweite Axes für die kinetische Energie.
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_title('Energieverteilung')
ax2.set_xlabel('m [kg]')
ax2.set_ylabel('E [J]')
ax2.set_ylim([0, E_max])
ax2.grid()

# Plotte die kinetische Energie als Funktion der Teilchenmasse.
plot, = ax2.plot(m, E_kin[0], 'o')

# Erzeuge einen Plot für die Ausgleichsgerade.
ausgleichsgerade, = ax2.plot(m, 0 * m, '-', linewidth=3)


def update(n):
    # Aktualisiere die Positionen der Teilchen.
    for i, k in enumerate(kreis):
        k.set_center(r[n, i])

    # Aktualisiere die Punkte für die Energieverteilung.
    plot.set_ydata(E_kin[n])

    # Aktualisiere die Ausgleichsgerade.
    steigung, yabschnitt = np.polyfit(m, E_kin[n], 1)
    ausgleichsgerade.set_ydata(yabschnitt + steigung * m)

    return kreis + [plot, ausgleichsgerade]


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
