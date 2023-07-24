"""Berechnung der Kraftverteilung und der Verformung einer Brücke
in linearer Näherung.

Die Visualisierung der Kräfte erfolgt mit einer Vektorpfeilen. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Lege einen Skalierungsfaktor für die Kraftvektoren fest.
scal_kraft = 0.0001

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

# Erdbeschleunigung [m/s²].
g = 9.81

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
F_ext = np.zeros((n_punkte, dim))
for i, stab in enumerate(staebe):
    for k in stab:
        F_ext[k, 1] -= laenge(i) * A * rho * g / 2


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

# Berechne die Kraft in jedem der Stäbe in linearer Näherung.
F = np.zeros(n_staebe)
for i, (j, k) in enumerate(staebe):
    ev = einheitsvektor(k, i)
    F[i] = S[i] / laenge(i) * ev @ (delta_r[j] - delta_r[k])

# Berechne die äußeren Kräfte.
for i, stab in enumerate(staebe):
    for k in np.intersect1d(stab, idx_stuetz):
        F_ext[k] -= F[i] * einheitsvektor(k, i)

# Berechne die neue Position der einzelnen Punkte.
punkte += delta_r

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-4, 24])
ax.set_ylim([-2, 4])
ax.set_aspect('equal')
ax.grid()

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[idx_knoten, 0], punkte[idx_knoten, 1], 'bo')
ax.plot(punkte[idx_stuetz, 0], punkte[idx_stuetz, 1], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0], punkte[stab, 1], color='black')
    # Erzeuge ein Textfeld, das den Wert der Kraft angibt. Das
    # Textfeld wird am Mittelpunkt des Stabes platziert.
    pos = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{kraft:+.1f} N', pos, color='blue')
    annot.draggable(True)

# Zeichne die äußeren Kräfte mit roten Pfeilen in das Diagramm
# ein und erzeuge Textfelder, die den Betrag der Kraft angeben.
style = mpl.patches.ArrowStyle.Simple(head_length=10,
                                      head_width=5)
for punkt, kraft in zip(punkte, F_ext):
    p1 = punkt + scal_kraft * kraft
    pfeil = mpl.patches.FancyArrowPatch(punkt, p1, color='red',
                                        arrowstyle=style,
                                        zorder=2)
    ax.add_artist(pfeil)
    annot = ax.annotate(f'{np.linalg.norm(kraft):.1f} N',
                        p1, color='red')
    annot.draggable(True)

# Zeichne die inneren Kräfte mit blauen Pfeilen in das Diagramm.
for i, stab in enumerate(staebe):
    for k in stab:
        r1 = punkte[k]
        r2 = r1 + einheitsvektor(k, i) * scal_kraft * F[i]
        pfeil1 = mpl.patches.FancyArrowPatch(r1, r2,
                                             color='blue',
                                             arrowstyle=style,
                                             zorder=2)
        ax.add_artist(pfeil1)

plt.show()
