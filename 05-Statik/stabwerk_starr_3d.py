"""Berechnung der Kraftverteilung in einem 3-dimensionalen
starren Stabwerk. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.005

# Anzahl der Raumdimensionen für das Problem (2 oder 3).
dim = 3

# Lege die Positionen der Punkte fest [m].
punkte = np.array([[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0],
                   [0, 0, 2], [1, 0, 2], [1, 1, 2],
                   [0, 1, 2]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
idx_stuetz = [0, 1, 2]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[0, 3], [1, 4], [2, 6],
                   [3, 4], [4, 5], [5, 6], [6, 3],
                   [3, 5], [0, 4],
                   [0, 6], [0, 5], [1, 5]])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 0, -98.1], [0, 0, -98.1], [0, 0, -98.1],
                  [0, 0, -98.1]])

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
ax = fig.add_subplot(1, 1, 1, projection='3d',
                     elev=40, azim=45)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.set_xlim([-0.5, 2.0])
ax.set_ylim([-0.5, 2.0])
ax.set_zlim([0, 3.0])
ax.grid()


# Definiere einen 3D-Pfeil.
class Arrow3D(mpl.patches.FancyArrowPatch):

    def __init__(self, posA, posB, *args, **kwargs):
        super().__init__(posA[0:2], posB[0:2], *args, **kwargs)
        self._pos = np.array([posA, posB])

    def set_positions(self, posA, posB):
        self._pos = np.array([posA, posB])

    def do_3d_projection(self, renderer=None):
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(*self._pos.T, self.axes.M)
        p = np.array(p)
        super().set_positions(p[:, 0], p[:, 1])
        return np.min(p[2, :])


class Annotation3D(mpl.text.Annotation):

    def __init__(self, s, pos, *args, **kwargs):
        super().__init__(s, xy=(0, 0), *args,
                         xytext=(0, 0),
                         textcoords='offset points',
                         **kwargs)
        self._pos = np.array(pos)

    def draw(self, renderer):
        p = mpl_toolkits.mplot3d.proj3d.proj_transform(
            *self._pos, self.axes.M)
        self.xy = p[0:2]
        super().draw(renderer)


# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[idx_knoten, 0],
        punkte[idx_knoten, 1],
        punkte[idx_knoten, 2], 'bo')
ax.plot(punkte[idx_stuetz, 0],
        punkte[idx_stuetz, 1],
        punkte[idx_stuetz, 2], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for stab, kraft in zip(staebe, F):
    ax.plot(punkte[stab, 0],
            punkte[stab, 1],
            punkte[stab, 2], color='black')
    # Erzeuge ein Textfeld, das den Wert der Kraft angibt. Das
    # Textfeld wird am Mittelpunkt des Stabes platziert.
    pos = np.mean(punkte[stab], axis=0)
    annot = Annotation3D(f'{kraft:+.1f} N', pos, color='blue')
    ax.add_artist(annot)
    annot.draggable(True)

# Zeichne die äußeren Kräfte mit roten Pfeilen in das Diagramm
# ein und erzeuge Textfelder, die den Betrag der Kraft angeben.
style = mpl.patches.ArrowStyle.Simple(head_length=10,
                                      head_width=5)
for punkt, kraft in zip(punkte, F_ext):
    p1 = punkt + scal_kraft * kraft
    pfeil = Arrow3D(punkt, p1, color='red',
                    arrowstyle=style, zorder=2)
    ax.add_artist(pfeil)
    annot = Annotation3D(f'{np.linalg.norm(kraft):.1f} N',
                         p1, color='red')
    ax.add_artist(annot)
    annot.draggable(True)

# Zeichne die inneren Kräfte mit blauen Pfeilen in das Diagramm.
for i, stab in enumerate(staebe):
    for k in stab:
        r1 = punkte[k]
        r2 = r1 + einheitsvektor(k, i) * scal_kraft * F[i]
        pfeil = Arrow3D(r1, r2, color='blue',
                        arrowstyle=style,
                        zorder=2)
        ax.add_artist(pfeil)

# Zeige die Grafik an.
plt.show()
