"""Animation: Schräger Stoß zweier kreisförmiger Objekte. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anzahl der Raumdimensionen.
dim = 2

# Simulationszeitdauer T und Schrittweite dt [s].
T = 8
dt = 0.02

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
v1 = np.array([1.0, 0.0])
v2 = np.array([0.0, 0.0])


def kollision(r1, r2, v1, v2):
    """Gibt die Zeit bis zur Teilchenkollision zurück. Wenn die
    Teilchen nicht kollidieren, wird NaN zurückgegeben. """

    # Differenz der Orts- und Geschwindigkeitsvektoren.
    dr = r1 - r2
    dv = v1 - v2

    # Um den Zeitpunkt der Kollision zu bestimmen, muss eine
    # quadratische Gleichung der Form
    #          t² + 2 a t + b = 0
    # gelöst werden. Nur die kleinere Lösung ist relevant.
    a = (dv @ dr) / (dv @ dv)
    b = (dr @ dr - (R1 + R2)**2) / (dv @ dv)
    D = a**2 - b
    t = -a - np.sqrt(D)

    return t


# Berechne Energie und Gesamtimpuls am Anfang.
E0 = 1/2 * m1 * v1 @ v1 + 1/2 * m2 * v2 @ v2
p0 = m1 * v1 + m2 * v2

# Lege Arrays für das Simulationsergebnis an.
t = np.arange(0, T, dt)
r1 = np.empty((t.size, dim))
r2 = np.empty((t.size, dim))

# Lege die Anfangsbedingungen fest.
r1[0] = r0_1
r2[0] = r0_2

# Berechne den Zeitpunkt der Kollision.
t_koll = kollision(r1[0], r2[0], v1, v2)

# Schleife der Simulation.
for i in range(1, t.size):
    # Kopiere die Positionen aus dem vorherigen Zeitschritt.
    r1[i] = r1[i - 1]
    r2[i] = r2[i - 1]

    # Kollidieren die Teilchen in diesem Zeitschritt?
    if t[i-1] < t_koll <= t[i]:
        # Bewege die Teilchen bis zum Kollisionszeitpunkt.
        r1[i] += v1 * (t_koll - t[i-1])
        r2[i] += v2 * (t_koll - t[i-1])

        # Berechne die Komponenten der Geschwindigkeiten
        # entlang der Verbindungslinie der Körper.
        er = (r1[i] - r2[i]) / np.linalg.norm(r1[i] - r2[i])
        v1_s = v1 @ er
        v2_s = v2 @ er

        # Benutze für diese Komponenten die Gleichungen für
        # den zentralen elastischen Stoß.
        v1_s_neu = (2 * m2 * v2_s + (m1 - m2) * v1_s) / (m1 + m2)
        v2_s_neu = (2 * m1 * v1_s + (m2 - m1) * v2_s) / (m1 + m2)

        # Ersetze diese Komponenten der Geschwindigkeiten.
        v1 += (v1_s_neu - v1_s) * er
        v2 += (v2_s_neu - v2_s) * er

        # Bewege die Teilchen bis zum Ende des Zeitschritts.
        r1[i] += v1 * (t[i] - t_koll)
        r2[i] += v2 * (t[i] - t_koll)
    else:
        # Bewege die Teilchen ohne Kollision.
        r1[i] += v1 * dt
        r2[i] += v2 * dt

# Berechne Energie und Gesamtimpuls am Ende.
E1 = 1/2 * m1 * v1 @ v1 + 1/2 * m2 * v2 @ v2
p1 = m1 * v1 + m2 * v2

# Gib die Energie und den Impuls vor und nach dem Stoß aus.
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
bahn1, = ax.plot([0], [0], '-r')
bahn2, = ax.plot([0], [0], '-b')

# Erzeuge zwei Kreise für die Darstellung der Körper.
kreis1 = mpl.patches.Circle([0, 0], R1, color='red')
kreis2 = mpl.patches.Circle([0, 0], R2, color='blue')
ax.add_artist(kreis1)
ax.add_artist(kreis2)


def update(n):
    # Aktualisiere die Position der beiden Körper.
    kreis1.set_center(r1[n])
    kreis2.set_center(r2[n])

    # Plotte die Bahnkurve bis zum aktuellen Zeitpunkt.
    bahn1.set_data(r1[:n, 0], r1[:n, 1])
    bahn2.set_data(r2[:n, 0], r2[:n, 1])
    return kreis1, kreis2, bahn1, bahn2


# Erstelle die Animation und starte sie.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
