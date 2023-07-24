"""Animation des keplerschen Flächensatzes. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Konstanten 1 Tag [s] und 1 Jahr [s].
tag = 24 * 60 * 60
jahr = 365.25 * tag

# Eine Astronomische Einheit [m].
AE = 1.495978707e11

# Simulationsdauer T und dargestellte Schrittweite dt [s].
T = 3 * jahr
dt = 0.5 * tag

# Anzahl der Zeitschritte, die für die Darstellung der Fläche
# des Fahrstrahls verwendet wird
dN = 40

# Massen der Sonne M [kg].
M = 1.9889e30

# Gravitationskonstante [m³ / (kg * s²)].
G = 6.6741e-11

# Anfangspositionen des Planeten [m].
r0 = np.array([152.10e9, 0.0])
v0 = np.array([0, 15e3])


def dgl(t, u):
    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.set_xlabel('x [AE]')
ax.set_ylabel('y [AE]')
ax.grid()

# Plotte die Bahnkurve des Himmelskörpers.
ax.plot(r[0] / AE, r[1] / AE, '-b')

# Erzeuge Punktplots, für die Positionen der Himmelskörper.
planet, = ax.plot([0], [0], 'o', color='red')
sonne, = ax.plot([0], [0], 'o', color='gold')

# Erzeuge ein Polygon zur Darstellung der überstrichenen Fläche
# und füge dieses der Axes hinzu.
flaeche = mpl.patches.Polygon([[0, 0], [0, 0]], closed=True,
                              alpha=0.5, facecolor='red')
ax.add_artist(flaeche)

# Erzeuge zwei Textfelder für die Angabe der verstrichenen Zeit
# und die berechnete Fläche.
text_t = ax.text(0.01, 0.95, '', color='black',
                 transform=ax.transAxes)
text_A = ax.text(0.01, 0.90, '', color='black',
                 transform=ax.transAxes)


def polygon_flaeche(x, y):
    """Berechne die Fläche eines Polygons mit den angegebenen
    Koordinaten mithilfe der gaußschen Trapezformel. """
    return 0.5 * abs((y + np.roll(y, 1)) @ (x - np.roll(x, 1)))


def update(n):
    # Aktualisiere die Position des Himmelskörpers.
    planet.set_data(r[:, n] / AE)

    # Aktualisiere des Polygon und die Angabe der Fläche.
    if n >= dN:
        # Erzeuge ein (dN + 2) x 2 - Array. Als ersten Punkt
        # enthält dies die Position (0, 0) der Sonne und die
        # weiteren Punkte sind die dN + 1 Punkte der Bahnkurve
        # des Planeten.
        xy = np.zeros((dN + 2, 2))
        xy[1:, :] = r[:, (n - dN):(n + 1)].T / AE
        flaeche.set_xy(xy)

        # Berechne die Fläche des Polygons und gebe diese aus.
        A = polygon_flaeche(xy[:, 0], xy[:, 1])
        text_A.set_text(f'A = {A:.2e} AE²')
    else:
        # Zu Beginn der Animation kann noch keine Fläche
        # dargestellt werden.
        flaeche.set_xy([[0, 0], [0, 0]])
        text_A.set_text(f'')

    # Aktualisiere das Textfeld für die Zeit.
    text_t.set_text(f't = {t[n] / tag:.0f} d')

    return planet, text_t, text_A, flaeche


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, frames=t.size)

plt.show()
