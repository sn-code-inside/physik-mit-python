"""Berechnung der Kraftverteilung und der Verformung einer
Brücke in linearer Näherung unter Belastung durch ein fahrendes
Fahrzeug. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import matplotlib.cm

# Lege die maximale Kraft für die Farbtabelle fest [N].
F_max = 70000

# Anzahl der Raumdimensionen für das Problem (2 oder 3).
dim = 2

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [4, 0], [8, 0],
                   [12, 0], [16, 0], [20, 0],
                   [2, 2], [6, 2], [10, 2],
                   [14, 2], [18, 2]], dtype=float)

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
idx_stuetz = [0, 5]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [6, 7], [7, 8], [8, 9], [9, 10],
                   [0, 6], [1, 7], [2, 8], [3, 9], [4, 10],
                   [6, 1], [7, 2], [8, 3], [9, 4], [10, 5]])

# Elastizitätsmodul [N/m²].
E = 210e9

# Querschnittsfläche der Stäbe [m²].
A = 5e-2 ** 2

# Dichte des Stabmaterials [kg/m³].
rho = 7860.0

# Masse des Fahrzeugs [kg].
masse = 3000.0

# Geschwindigkeit des Fahrzeugs [m/s].
v = 0.03

# Zeitdauer zwischen zwei Bildern.
dt = 0.5

# Erdbeschleunigung [m/s²].
g = 9.81

# Länge eines Segments der Fahrbahn [m].
L_segment = 4.0

# Gesamtlänger der Brücke [m].
L_gesamt = 20.0

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.zeros((11, 2))

# Definiere die Anzahl der Punkte, Stäbe und Stützpunkte.
n_punkte = punkte.shape[0]
n_staebe = staebe.shape[0]
n_stuetzpunkte = len(idx_stuetz)
n_knoten = n_punkte - n_stuetzpunkte
n_gleichungen = n_knoten * dim

# Lege die Steifigkeit jedes Stabes fest.
S = np.ones(n_staebe) * E * A


# Erzeuge eine Liste mit den Indizes der Knoten.
idx_knoten = list(set(range(n_punkte)) - set(idx_stuetz))


def einheitsvektor(i_punkt, i_stab):
    """Gibt den Einheitsvektor zurück, der vom Punkt i_punkt
    entlang des Stabes Index i_stab zeigt. """
    i1, i2 = staebe[i_stab]
    if i_punkt == i1:
        vec = punkte[i2] - punkte[i1]
    else:
        vec = punkte[i1] - punkte[i2]
    return vec / np.linalg.norm(vec)


def laenge(i_stab):
    """Gibt die Länge des Stabes i_stab zurück. """
    i1, i2 = staebe[i_stab]
    return np.linalg.norm(punkte[i2] - punkte[i1])


# Lege die äußere Kraft auf jeden Knotenpunkt durch die
# Gewichtskraft der angrenzenden Stäbe fest:
F_ext0 = np.zeros((n_punkte, dim))
for i, stab in enumerate(staebe):
    for k in stab:
        F_ext0[k, 1] -= laenge(i) * A * rho * g / 2

# Stelle das Gleichungssystem für die Kräfte auf.
A = np.zeros((n_gleichungen, n_gleichungen))
for i, stab in enumerate(staebe):
    for k in np.intersect1d(stab, idx_knoten):
        n = idx_knoten.index(k)
        for j in np.intersect1d(stab, idx_knoten):
            m = idx_knoten.index(j)
            ee = np.outer(einheitsvektor(k, i),
                          einheitsvektor(j, i))
            A[n * dim:(n + 1) * dim,
              m * dim:(m + 1) * dim] += - S[i] * ee / laenge(i)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-2, 22])
ax.set_ylim([-2, 4])
ax.set_aspect('equal')
ax.grid()

# Erzeuge ein Objekt, das jeder Kraft eine Farbe zuordnet.
mapper = mpl.cm.ScalarMappable(cmap=matplotlib.cm.jet)
mapper.set_array([-F_max, F_max])
mapper.autoscale()

# Erzeuge einen Farbbalken am Rand des Bildes.
fig.colorbar(mapper, format='%.0g', label='Kraft [N]', pad=0.2,
             fraction=0.12, orientation='horizontal', ax=ax)

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
plt_knot, = ax.plot(punkte[idx_knoten, 0],
                    punkte[idx_knoten, 1], 'bo', zorder=3)
plt_stuetz, = ax.plot(punkte[idx_stuetz, 0],
                      punkte[idx_stuetz, 1], 'ro', zorder=3)

# Plotte das Fahrzeug in Grün.
plt_fahrz, = ax.plot(0, 0, 'go', zorder=4)

# Plotte die Stäbe und wähle die Farbe entsprechend der
# wirkenden Kraft.
plt_stab = []
for stab in staebe:
    p, = ax.plot(punkte[stab, 0], punkte[stab, 1], linewidth=3)
    plt_stab.append(p)


def update(frame):

    # Lege die aktuelle Position des Autos fest und ermittle
    # einen Index 'ind', bei die Gewichtskraft des Autos
    # mit dem Anteil 'frac' berücksichtigt werden
    # muss. Der Rest der Gewichtskraft muss beim nächsten
    # Index berücksichtigt werden.
    pos = v * dt * frame
    frac, ind = math.modf(pos / L_segment)
    ind = int(ind)

    # Berücksichtige die Gewichtskraft des Autos
    F_ext = F_ext0.copy()
    F_ext[ind, 1] -= masse * g * (1 - frac)
    F_ext[ind+1, 1] -= masse * g * frac

    # Löse das Gleichungssystem A @ dr = -F_ext.
    dr = np.linalg.solve(A, -F_ext[idx_knoten].reshape(-1))
    dr = dr.reshape(n_knoten, dim)

    # Das Array dr enthält nur die Verschiebungen der
    # Knotenpunkte. Für den weiteren Ablauf des Programms ist es
    # praktisch, stattdessen ein Array zu haben, das die gleiche
    # Größe, wie das Array 'punkte' hat und an den Stützstellen
    # Nullen enthält.
    delta_r = np.zeros((n_punkte, dim))
    delta_r[idx_knoten] = dr

    # Berechne die neue Position der einzelnen Punkte.
    punkte1 = punkte + delta_r

    # Berechne die Kraft in jedem der Stäbe in linearer Näherung.
    F = np.zeros(n_staebe)
    for i, (j, k) in enumerate(staebe):
        ev = einheitsvektor(k, i)
        F[i] = S[i] / laenge(i) * ev @ (delta_r[j] - delta_r[k])

    # Berechne die äußeren Kräfte.
    for i, stab in enumerate(staebe):
        for k in np.intersect1d(stab, idx_stuetz):
            F_ext[k] -= F[i] * einheitsvektor(k, i)

    # Aktualisiere die Postion der Knotenpunkte.
    plt_knot.set_data(punkte1[idx_knoten, 0],
                      punkte1[idx_knoten, 1])

    # Aktualisiere die Position der Stäbe.
    for p, stab in zip(plt_stab, staebe):
        p.set_data(punkte1[stab, 0], punkte1[stab, 1])

    # Aktualisiere die Farbe der Stäbe.
    for p, kraft in zip(plt_stab, F):
        p.set_color(mapper.to_rgba(kraft))

    # Aktualisiere die Position des Fahrzeugs.
    plt_fahrz.set_xdata([pos])

    return [plt_fahrz, plt_knot, plt_stuetz] + plt_stab


# Berechne die Anzahl von Frames, in denen das Fahrzeug
# die Brücke komplett überquert.
n_frames = int(L_gesamt / (v * dt))

# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  frames=n_frames,
                                  interval=30, blit=True)

# Starte die Animation.
plt.show()
