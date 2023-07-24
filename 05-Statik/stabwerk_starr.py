"""Berechnung der Kraftverteilung in einem 2-dimensionalen
starren Stabwerk. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.002

# Anzahl der Raumdimensionen für das Problem (2 oder 3).
dim = 2

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0], [0, 1.1], [1.2, 0], [2.5, 1.1]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
idx_stuetz = [0, 1]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 2], [1, 2], [2, 3], [1, 3]])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.array([[0, 0], [0, 0], [0, -147.15], [0, -98.1]])

# Definiere die Anzahl der Punkte, Stäbe etc.
n_punkte = punkte.shape[0]
n_staebe = staebe.shape[0]
n_stuetz = len(idx_stuetz)
n_knoten = n_punkte - n_stuetz
n_gleichungen = n_knoten * dim

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
ax.set_xlim([-1.0, 3.0])
ax.set_ylim([-0.5, 2.0])
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
        pfeil = mpl.patches.FancyArrowPatch(r1, r2, color='blue',
                                            arrowstyle=style,
                                            zorder=2)
        ax.add_artist(pfeil)

# Zeige die Grafik an.
plt.show()
