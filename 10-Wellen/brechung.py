"""Animation zum Brechungs- und Reflextionsgesetz mit dem
huygensschen Prinzip. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Ausbreitungsgeschwindigkeit in den beiden Medien [m/s].
c1 = 1.0
c2 = 0.3

# Einfallswinkel [rad].
alpha = np.radians(50)

# Austrittswinkel der gebrochenen Strahlen nach Snellius [rad].
beta = np.arcsin(np.sin(alpha) * c2 / c1)

# Dargestellter Ortsbereich -xlim ... xlim, -ylim ... ylim [m].
xlim = 15
ylim = 15

# Zeitschrittweite [s].
dt = 0.05

# Anzahl der Strahlen.
n_strahlen = 7

# Breite des einfallenden Strahlenbündes [m].
breite = 10.0

# Anfangsabstand des Mittelstrahls von der Grenzfläche [m].
dist0 = 21.0

# Berechne die Startpunkte r_start der einzelnen Strahlen.
b = np.linspace(-breite / 2, breite / 2, n_strahlen)
r_start = np.empty((n_strahlen, 2))
r_start[:, 0] = -dist0 * np.sin(alpha) + b * np.cos(alpha)
r_start[:, 1] = dist0 * np.cos(alpha) + b * np.sin(alpha)

# Berechne die Auftreffzeitpunkte der einzelnen Strahlen.
t_auf = (dist0 + b * np.tan(alpha)) / c1

# Richtungsvektor der einfallenden Strahlen, reflektierten und
# der gebrochenen Strahlen.
e1 = np.array([np.sin(alpha), -np.cos(alpha)])
er = np.array([np.sin(alpha), np.cos(alpha)])
et = np.array([np.sin(beta), -np.cos(beta)])

# Berechne die Auftreffzeitpunkte der Strahlen.
r_auf = r_start + t_auf.reshape(-1, 1) * c1 * e1

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-xlim, xlim)
ax.set_ylim(-ylim, ylim)
ax.set_aspect('equal')

# Erzeuge leere Listen für die einfallenden, reflektierten und
# transmittierten Strahlen sowie für die Kreisbögen der
# Elementarwellen.
strahl_1, strahl_r, strahl_t = [], [], []
kreis_r, kreis_t = [], []

# Erzeuge die entsprechenden Grafikelemente und füge sie den
# Listen hinzu.
for i in range(n_strahlen):
    # Erzeuge einen einfallenden Lichtstrahl.
    l, = ax.plot([0], [0])
    farbe = l.get_color()
    strahl_1.append(l)

    # Erzeuge den reflektierten und transmittierten Lichtstrahl
    # mit der gleichen Farbe, wie der entsprechende einfallende
    # Lichtstrahl.
    lr, = ax.plot([0], [0], ':', color=farbe)
    lt, = ax.plot([0], [0], '--', color=farbe)
    strahl_r.append(lr)
    strahl_t.append(lt)

    # Erzeuge die Kreisbögen für die Elementarwellen.
    kr = mpl.patches.Arc(r_auf[i], width=1, height=1,
                         theta1=0, theta2=180,
                         fill=False, color=farbe)
    kt = mpl.patches.Arc(r_auf[i], width=1, height=1,
                         theta1=180, theta2=360,
                         fill=False, color=farbe)
    ax.add_artist(kr)
    ax.add_artist(kt)
    kreis_r.append(kr)
    kreis_t.append(kt)

# Färbe die obere Hälfte des Koordinatensystems hellgrau ein.
rec = mpl.patches.Rectangle((-xlim, 0),2 * xlim, ylim,
                            color='0.9', zorder=0)
ax.add_artist(rec)

# Fäbe die untere Hälfte des Koordinatensystem etwas dunkler ein.
rec2 = mpl.patches.Rectangle((-xlim, -ylim), 2 * xlim, ylim,
                             color='0.8', zorder=0)
ax.add_artist(rec2)


def update(n):
    # Berechne den aktuellen Zeitpunkt.
    t = dt * n

    # Aktualisiere die einfallenden Strahlen.
    for strahl1, r0, ta in zip(strahl_1, r_start, t_auf):
        p = np.array([r0, r0 + c1 * min(t, ta) * e1])
        strahl1.set_data(p.T)

    # Aktualisiere die reflektierten Strahlen.
    for strahl, ra, ta in zip(strahl_r, r_auf, t_auf):
        t1 = max(0, (t - ta))
        p = np.array([ra, ra + c1 * t1 * er])
        strahl.set_data(p.T)

    # Aktualisiere die transmittierten Strahlen.
    for strahl, ra, ta in zip(strahl_t, r_auf, t_auf):
        t1 = max(0, (t - ta))
        p = np.array([ra, ra + c2 * t1 * et])
        strahl.set_data(p.T)

    # Aktualisiere die Kreise.
    for k1, k2, ta in zip(kreis_r, kreis_t, t_auf):
        if t > ta:
            k1.width = k1.height = 2 * (t - ta) * c1
            k2.width = k2.height = 2 * (t - ta) * c2
        k1.set_visible(t > ta)
        k2.set_visible(t > ta)

    return (strahl_1 + strahl_r + strahl_t +
            kreis_r + kreis_t)


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)
plt.show()
