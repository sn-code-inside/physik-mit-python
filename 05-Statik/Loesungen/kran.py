"""Berechnung der Kraftverteilung in einem 2-dimensionalen
Kranausleger, der aus starren Stäben besteht. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm

# Lege die maximale Kraft für die Farbtabelle fest [N].
F_max = 9000

# Anzahl der Raumdimensionen für das Problem (2 oder 3).
dim = 2

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [0.7, 0], [2.1, 0], [3.5, 0],
                   [4.9, 0], [6.3, 0], [0, 1], [1.4, 1],
                   [2.8, 1], [4.2, 1], [5.6, 1]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
idx_stuetz = [0, 6]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
                   [6, 7], [7, 8], [8, 9], [9, 10],
                   [6, 1], [1, 7], [7, 2], [2, 8], [8, 3],
                   [3, 9], [9, 4], [4, 10], [10, 5]])

# Definiere die Anzahl der Punkte, Stäbe etc.
n_punkte = punkte.shape[0]
n_staebe = staebe.shape[0]
n_stuetz = len(idx_stuetz)
n_knoten = n_punkte - n_stuetz
n_gleichungen = n_knoten * dim

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.zeros((n_punkte, dim))
F_ext[:, 1] = -100.0
F_ext[5, 1] = -1000.0
F_ext[idx_stuetz] = 0

# Erzeuge eine Liste mit den Indizes der Knoten.
idx_knoten = list(set(range(n_punkte)) - set(idx_stuetz))


def einheitsvektor(i_punkt, i_stab):
    """Gibt den Einheitsvektor zurück, der vom Punkt i_punkt
    entlang des Stabes mit dem Index i_stab zeigt. """
    i1, i2 = staebe[i_stab]
    if i_punkt == i1:
        vec = punkte[i2] - punkte[i1]
    else:
        vec = punkte[i1] - punkte[i2]
    return vec / np.linalg.norm(vec)


# Stelle das Gleichungssystem für die Kräfte auf.
A = np.zeros((n_gleichungen, n_gleichungen))
for i, stab in enumerate(staebe):
    for k in np.intersect1d(stab, idx_knoten):
        n = idx_knoten.index(k)
        A[n * dim:(n + 1) * dim, i] = einheitsvektor(k, i)

# Löse das Gleichungssystem A @ F = -F_ext nach den Kräften F.
b = -F_ext[idx_knoten].reshape(-1)
F = np.linalg.solve(A, b)

# Berechne die äußeren Kräfte.
for i, stab in enumerate(staebe):
    for k in np.intersect1d(stab, idx_stuetz):
        F_ext[k] -= F[i] * einheitsvektor(k, i)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-1.0, 7.0])
ax.set_ylim([-0.5, 2.0])
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen Mapper und einen Farbbalken.
mapper = mpl.cm.ScalarMappable(cmap=mpl.cm.jet)
mapper.set_clim([-F_max, F_max])
mapper.set_array([])
fig.colorbar(mapper, format='%.0g', label='Kraft [N]',
             orientation='horizontal', shrink=0.5)

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[idx_knoten, 0], punkte[idx_knoten, 1], 'bo')
ax.plot(punkte[idx_stuetz, 0], punkte[idx_stuetz, 1], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0], punkte[stab, 1], linewidth=2,
            color=mapper.to_rgba(kraft))
    # Erzeuge ein Textfeld, das den Wert der Kraft angibt. Das
    # Textfeld wird am Mittelpunkt des Stabes platziert.
    pos = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{kraft:+.1f} N', pos, color='blue')
    annot.draggable(True)

# Zeige die Grafik an.
plt.show()
