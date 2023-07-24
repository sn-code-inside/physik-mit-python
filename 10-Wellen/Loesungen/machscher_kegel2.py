"""Visualisierung für das doppelte Eintreffen von Wellen
innerhalb des machschen Kegels. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Zeitschrittweite [s].
dt = 0.1

# Dargestellter Ortsbereich 0..+xmax, 0..+ymax [m].
xlim = (-2.0, 2.0)
ylim = (-1.0, 1.0)

# Frequenz, mit der die Wellenzüge aussendet werden [Hz].
f_Q = 5.0

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 1.0

# Lege die Startposition der Quelle und des Beobachters fest [m].
r0_Q = np.array([-30.0, 0.0])
r0_B = np.array([0, -0.5])

# Lege die Geschwindigkeit von Quelle und Beobachter fest [m/s].
v_Q = np.array([1.3, 0.0])
v_B = np.array([0, 0])

# Berechne den Öffnungwinkel des machschen Kegels nach der
# gegebenen Gleichung.
phi = 2 * np.arcsin(c / np.linalg.norm(v_Q))

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

# Erzeuge zwei Kreise für Quelle und Beobachter.
quelle = mpl.patches.Circle((0, 0), radius=0.03,
                            color='black', fill=True, zorder=4)
beobach = mpl.patches.Circle((0, 0), radius=0.03,
                             color='blue', fill=True, zorder=4)
ax.add_patch(quelle)
ax.add_patch(beobach)

# Erzeuge einen Linienplot für den machschen Kegel. Der Kegel
# wird später in der update-Funktion durch drei Punkte
# dargestellt.
kegel, = ax.plot([0], [0], 'k', linewidth=2, zorder=5)

# Beschriftung.
ax.text(0.05, 0.95, f'$v_Q = {np.linalg.norm(v_Q) / c:.2f} c$',
        transform=ax.transAxes, color='black', size=12,
        horizontalalignment='left', verticalalignment='top')
ax.text(0.95, 0.95, f'$v_B = {np.linalg.norm(v_B) / c:.2f} c$',
        transform=ax.transAxes, color='blue', size=12,
        horizontalalignment='right', verticalalignment='top')

# Lege eine Liste an, die die kreisförmigen Wellenzüge speichert.
kreise = []


def update(n):
    # Berechne den aktuellen Zeitpunkt.
    t = dt * n

    # Berechne die aktuelle Position von Quelle und Beobachter.
    quelle.center= r0_Q + v_Q * t
    beobach.center = r0_B + v_B * t

    # Zeichne den Kegel. Die Spitze des Kegels soll sich an
    # der aktuellen Position des Senders befinden. Wir müssen
    # noch zwei weitere Punkte r1 und r2 konstruieren,
    # die sich in Bewegungsrichtung unter einem Winkel phi/2
    # befinden. Dazu erzeugen wir zunächst einen
    # Einheitsvektor e_v in Beweungsrichtung und einen
    # Einheitsvektor e_s der senkrecht dazu steht.
    e_v = v_Q / np.linalg.norm(v_Q)
    e_s = np.array([e_v[1], -e_v[0]])

    # Jetzt konstruieren wir jeweils einen Richtungsvektor
    # 'e_dir' der einen Winkel phi/2 mit der Bewegungsrichtung
    # einschließt und berechnen die zusätzlichen Punkte über
    # eine Geradengleichung. Wir wählen die Länge L der Linien
    # dabei so lang, dass diese gerade bis zum ersten
    # ausgesendeten Wellenzug reichen.
    L = t * np.sqrt(v_Q @ v_Q - c ** 2)
    e_dir = -np.cos(phi / 2) * e_v + np.sin(phi / 2) * e_s
    r1 = quelle.center + L * e_dir
    e_dir = -np.cos(phi / 2) * e_v - np.sin(phi / 2) * e_s
    r2 = quelle.center + L * e_dir

    # Setze die drei Punkte zusammen und aktualisiere den Plot.
    dat = np.stack([r1, quelle.center, r2])
    kegel.set_data(dat.T)

    # Erzeuge zum Startzeitpunkt einen neuen Kreis oder wenn
    # seit dem Aussenden des letzten Wellenzuges eine
    # Periodendauer vergangen ist.
    if not kreise or t >= kreise[-1].startzeit + 1 / f_Q:
        kreis = mpl.patches.Circle(quelle.center, radius=0,
                                   color='red', linewidth=1.5,
                                   fill=False, zorder=3)
        kreis.startzeit = t
        kreise.append(kreis)
        ax.add_patch(kreis)

    # Aktualisiere die Radien aller dargestellen Kreise.
    for kreis in kreise:
        kreis.radius = (t - kreis.startzeit) * c

    # Färbe den Beobachter rot, wenn ein Wellenzug auftrifft.
    beobach.set_color('blue')
    for kreis in kreise:
        d = np.linalg.norm(kreis.center - beobach.center)
        if abs(d - kreis.radius) < beobach.radius:
            beobach.set_color('red')

    # Färbe den Quelle rot, wenn ein Wellezug augesendet wird.
    d = np.linalg.norm(kreise[-1].center - quelle.center)
    if abs(d - kreise[-1].radius) < quelle.radius:
        quelle.set_color('red')
    else:
        quelle.set_color('black')

    return kreise + [quelle, beobach, kegel]


# Erezeuge ein einzelnes Bild zum n-ten Zeitschritt. Damit die
# Kreise korrekt erzeugt werden, müssen dafür die
# vorausgehenden Zeitschritte ebenfalls berechnet werden.
n = 240
for k in range(n):
    update(k)
plt.show()
