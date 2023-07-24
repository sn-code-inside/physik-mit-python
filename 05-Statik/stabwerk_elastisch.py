"""Berechnung der Kraftverteilung und der Verformung eines
2-dimensionalen elastischen Stabwerks. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.optimize

# Lege den Skalierungsfaktor für die Kraftvektoren fest [m/N].
scal_kraft = 0.001

# Anzahl der Raumdimensionen für das Problem (2 oder 3).
dim = 2

# Lege die Positionen der Punkte fest [m].
punkte0 = np.array([[0, 0], [1.2, 0], [1.2, 2.1], [0, 2.1],
                    [0.6, 1.05]])

# Erzeuge eine Liste mit den Indizes der Stützpunkte.
idx_stuetz = [0, 1]

# Jeder Stab verbindet genau zwei Punkte. Wir legen dazu die
# Indizes der zugehörigen Punkte in einem Array ab.
staebe = np.array([[1, 2], [2, 3], [3, 0],
                   [0, 4], [1, 4], [3, 4], [2, 4]])

# Produkt aus E-Modul und Querschnittsfläche der Stäbe [N].
S = np.array([5.6e6, 5.6e6, 5.6e6, 7.1e3, 7.1e3, 7.1e3, 7.1e3])

# Lege die äußere Kraft fest, die auf jeden Punkt wirkt [N].
# Für die Stützpunkte setzen wir diese Kraft zunächst auf 0.
# Diese wird später berechnet.
F_ext = np.array([[0, 0], [0, 0], [200.0, 0], [0, 0], [0, 0]])

# Definiere die Anzahl der Punkte, Stäbe und Stützpunkte.
n_punkte = punkte0.shape[0]
n_staebe = staebe.shape[0]
n_stuetzpunkte = len(idx_stuetz)
n_knoten = n_punkte - n_stuetzpunkte

# Erzeuge eine Liste mit den Indizes der Knoten.
idx_knoten = list(set(range(n_punkte)) - set(idx_stuetz))


def einheitsvektor(p, i_punkt, i_stab):
    """Gibt den Einheitsvektor zurück, der vom Punkt i_punkt
    entlang des Stabes mit dem Index i_stab zeigt. """
    i1, i2 = staebe[i_stab]
    if i_punkt == i1:
        vec = p[i2] - p[i1]
    else:
        vec = p[i1] - p[i2]
    return vec / np.linalg.norm(vec)


def laenge(p, i_stab):
    """Gibt die Länge des Stabes i_stab zurück. """
    i1, i2 = staebe[i_stab]
    return np.linalg.norm(p[i2] - p[i1])


def stabkraft(p, i):
    """Berechnet die Kraft im Stab i für die Punkte p. """
    l0 = laenge(punkte0, i)
    return S[i] * (laenge(p, i) - l0) / l0


def gesamtkraft(p):
    """Gibt die Gesamtkraft für jeden der Punkte p zurück. """

    # Initialisiere das Array mit den äußeren Kräften.
    F_ges = F_ext.copy()

    # Addiere für jeden Stab die Stabkraft für die angrenzenden
    # Punkte.
    for i, stab in enumerate(staebe):
        for k in stab:
            F_ges[k] += stabkraft(p, i) * einheitsvektor(p, k, i)

    return F_ges


# Für die Lösung des Problems definieren wir eine Funktion
# func, die als Variable nur die Positionen der Knoten x
# übergeben bekommt und die die Kraft als ein 1-dimensionales
# Array zurückgibt. Wir suchen dann die Positionen der Knoten
# so, dass func(x) = 0 ist.
def func(x):
    # Erzeuge ein Array, das die Positionen der Stützpunkte
    # enthält und die Positionen x der Knotenpunkte.
    p = punkte0.copy()
    p[idx_knoten] = x.reshape(n_knoten, dim)

    # Berechne die Gesamtkraft für jeden einzelnen Punkt.
    F_ges = gesamtkraft(p)

    # Wähle die Knotenkräfte aus.
    F_knoten = F_ges[idx_knoten]

    # Gib das Ergebnis als 1-dimensionales Array zurück.
    return F_knoten.reshape(-1)


# Suche eine Lösung der Gleichung func(x) = 0. Als
# Startpositionen geben wir die Anfangspositonen der Knoten vor.
result = scipy.optimize.root(func, punkte0[idx_knoten])
print(result.message)
print(f'Die Funktion wurde {result.nfev}-mal ausgewertet.')

# Erzeuge ein Array mit den berechneten Positonen der Punkte.
punkte = punkte0.copy()
punkte[idx_knoten] = result.x.reshape(n_knoten, dim)

# Berechne die Kraft in jedem der Stäbe.
F = np.zeros(n_staebe)
for i in range(n_staebe):
    F[i] = stabkraft(punkte, i)

# Berechne die äußeren Kräfte auf die Stützpunkte.
F_ext[idx_stuetz] = -gesamtkraft(punkte)[idx_stuetz]

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim([-0.3, 1.8])
ax.set_ylim([-0.5, 2.5])
ax.set_aspect('equal')
ax.grid()

# Plotte die Knotenpunkte in Blau und die Stützpunkte in Rot.
ax.plot(punkte[idx_knoten, 0], punkte[idx_knoten, 1], 'bo')
ax.plot(punkte[idx_stuetz, 0], punkte[idx_stuetz, 1], 'ro')

# Plotte die Stäbe und beschrifte diese mit dem Wert der Kraft.
for i, stab in enumerate(staebe):
    ax.plot(punkte[stab, 0], punkte[stab, 1], color='black')
    # Erzeuge ein Textfeld, das den Wert der Kraft angibt. Das
    # Textfeld wird am Mittelpunkt des Stabes platziert.
    pos = np.mean(punkte[stab], axis=0)
    annot = ax.annotate(f'{F[i]:+.1f} N', pos, color='blue')
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
        r2 = r1 + einheitsvektor(punkte, k, i) * scal_kraft * F[i]
        pfeil = mpl.patches.FancyArrowPatch(r1, r2, color='blue',
                                            arrowstyle=style,
                                            zorder=2)
        ax.add_artist(pfeil)

# Zeige die Grafik an.
plt.show()
