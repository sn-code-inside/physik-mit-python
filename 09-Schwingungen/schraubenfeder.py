"""Darstellung einer Schraubenfeder (2-dimensional). """

import numpy as np
import matplotlib.pyplot as plt


def data(r0, r1, N=5, a=0.1, L0=0, R0=0.2, Np=300):
    """Berechne die Daten für die Darstellung einer
    Schraubenfeder in Form eines 2 x Np  - Arrays.

        r0 : Vektor des Anfangspunktes.
        r1 : Vektor des Endpunktes.
        N  : Anzahl der Windungen.
        a  : Länge der geraden Anfangs- und Endstücke.
        L0 : Ruhelänge der Feder. Wenn L0 > 0 ist, dann wird
             der Radius der Feder automatisch angepasst, wobei
             eine konstante Drahtlänge angenommen wird.
        R0 : Radius der Feder bei der Ruhelänge L0.
        Np : Anzahl der zu berechnenden Punkte. """

    # Gesamtlänge der Feder.
    L = np.linalg.norm(r1 - r0)

    # Passe den Radius der Feder an, sodass die Gesamtlänge
    # des Drahtes unverändert bleibt.
    l = np.sqrt((L0 - 2 * a) ** 2 + (2 * np.pi * N * R0) ** 2)
    if L - 2 * a < l:
        R = np.sqrt(l**2 - (L - 2 * a)**2) / (2 * np.pi * N)
    else:
        R = 0

    # Für L0 <= 0 erfolgt keine Anpassung des Radius.
    if L0 <= 0:
        R = R0

    # Array für das Ergebnis.
    dat = np.empty((Np, 2))

    # Setze den Anfangs- und Endpunkt.
    dat[0] = r0
    dat[-1] = r1

    # Einheitsvektor in Richtung der Verbindungslinie.
    er = (r1 - r0) / L

    # Einheitsvektor senkrecht zur Verbindungslinie.
    es = np.array([er[1], -er[0]])

    # Parameter entlang der Feder.
    s = np.linspace(0, 1, Np-2)
    s = s.reshape(-1, 1)

    # Berechne die restlichen Punkte als sinusförmige Linie.
    dat[1:-1] = r0 + er * (a + s * (L - 2 * a)) + es * (
                R * np.sin(N * 2 * np.pi * s))

    return dat.T


# Der folgende Code wird nur aufgerufen, wenn diese Modul
# direkt in Python gestartet wird. Wenn das Modul von einem
# anderen Python-Modul importiert wird, dann wird dieser Code
# nicht ausgeführt.
if __name__ == '__main__':

    # Erzeuge die Daten für die Schraubenfeder.
    r0 = np.array([0, 2.0])
    r1 = np.array([0, -1.0])
    dat = data(r0, r1, L0=3)

    # Erzeuge eine Figure und ein Axes-Objekt.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    # Stelle die Schraubenfeder mit einen Linienplot dar.
    plot, = ax.plot(dat[0], dat[1], 'k-', linewidth=2)
    plt.show()
