"""Simulation zweier gekoppelter Masse-Feder-Schwinger. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate
import schraubenfeder

# Simulationszeit T und Zeitschrittweite dt [s].
T = 42.0
dt = 0.01

# Masse der Körpers [kg].
m = 1.0

# Federkonstante [N/m].
D = 35.808088530029416

# Federkostante der Kopplungsfeder [N/m].
D1 = 1.8351645371640082

# Anfangslänge der äußeren Federn [m].
L = 0.3

# Anfangslänge der Kopplungsfeder [m].
L1 = 0.5

# Anfangsauslenkungen [m].
x0 = np.array([0.1, 0.0])

# Anfangsgeschwindigkeiten [m/s].
v0 = np.array([0.0, 0.0])


def dgl(t, u):
    x1, x2, v1, v2 = u
    F1 = -D * x1 - D1 * (x1 - x2)
    F2 = -D * x2 - D1 * (x2 - x1)
    return np.array([v1, v2, F1 / m, F2 / m])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((x0, v0))

# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
x1, x2, v1, v2 = result.y

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Erzeuge einen Plot der Auslenkung als Funktion der Zeit.
ax1 = fig.add_subplot(2, 1, 2)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('x [m]')
ax1.set_xlim(0, T)
ax1.grid()
ax1.plot(t, x1, 'b-', label='x1')
ax1.plot(t, x2, 'r-', label='x2')
ax1.legend()

# Linie zur Darstellung der aktuellen Zeit.
linie_t, = ax1.plot(0, 0, '-k', linewidth=3, zorder=5)

# Erzeuge einen Plot zur animierten Darstellung.
ax = fig.add_subplot(2, 1, 1)
ax.set_xlim([-L/20, 2 * L + L1 + L/20])
ax.set_aspect('equal')
ax.set_xlabel('x [m]')

# Mache die überflüssigen Achsenmarkierungen unsichtbar.
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_yaxis().set_visible(False)

# Linke Befestigung.
ax.plot([0, 0], [-L/4, L/4], 'k-', linewidth=3, zorder=6)

# Rechte Befestigung.
ax.plot([2 * L + L1, 2 * L + L1], [-L/4, L/4], 'k-',
        linewidth=3, zorder=6)

# Ruhelagen der beiden Massen.
ax.plot([L, L], [-L/4, L/4], '--',
        color='gray', zorder=3)
ax.plot([L + L1, L + L1], [-L/4, L/4], '--',
        color='gray', zorder=3)


# Die beiden Massenpunkte.
masse1, = ax.plot(0, 0, 'bo', zorder=5)
masse2, = ax.plot(0, 0, 'ro', zorder=5)

# Die drei Federn.
feder1, = ax.plot([0], [0], 'k-', zorder=4)
feder2, = ax.plot([0], [0], 'k-', zorder=4)
federk, = ax.plot([0], [0], 'g-', zorder=4)

# Textfeld zur Anzeige des Aktuellen Zeitpunkts.
text = ax.text(0.5, 0.8, '', transform=ax.transAxes)


def update(n):
    # Aktuelle Position der beiden Massen:
    r1 = np.array([L + x1[n], 0])
    r2 = np.array([L + L1 + x2[n], 0])

    # Position der beiden Befestigungspunkte:
    ra = np.array([0.0, 0.0])
    rb = np.array([2 * L + L1, 0.0])

    # Aktulisiere die Position der Massen.
    masse1.set_data(r1)
    masse2.set_data(r2)

    # Aktualisiere die linke Feder.
    feder1.set_data(schraubenfeder.data(ra, r1, N=10, R0=L/20,
                                        a=L/10, L0=L))

    # Aktualisiere die mittlere Kopplungsfeder.
    federk.set_data(schraubenfeder.data(r1, r2, N=10, R0=L/20,
                                        a=L/10, L0=L))

    # Aktualisiere die rechte Feder.
    feder2.set_data(schraubenfeder.data(r2, rb, N=10, R0=L/20,
                                        a=L/10, L0=L))

    # Aktualisiere die Zeitanzeige.
    text.set_text(f't = {t[n]:5.1f} s')

    # Aktualisiere die Markierung der aktuellen Zeit.
    t_akt = t[n]
    linie_t.set_data([[t_akt, t_akt], ax1.get_ylim()])

    return (masse1, masse2, feder1, feder2,
            federk, text, linie_t)


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
