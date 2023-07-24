"""Wellenausbreitung auf einer gespannten Masse-Feder-Kette.

In diesem Beispiel wird nur eine Transversalwelle angeregt,
wobei ein einzelner relativ langer Anregungspuls mit großer
Amplitude verwendet wird. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Simulationsdauer [s].
T = 10.0

# Zeitschrittweite [s].
dt = 0.01

# Dimension des Raumes.
dim = 2

# Anzahl der Massen.
N = 70

# Federkonstante [N/m].
D = 100

# Masse [kg].
m = 0.05

# Länge der ungespannten Federn [m].
L0 = 0.05

# Länge der Federn im gespannten Zustand [m].
L = 0.15

# Amplitude der longitudinalen und transversalen Anregung [m].
A_long = 0.0                           # keine Longitudinalwelle.
A_tran = 0.5                     # Große Amplitude (transversal).

# Ruhelage der N Massen im Abstand L auf der x-Achse.
r0 = np.zeros((N, dim))
r0[:, 0] = np.linspace(L, N * L, N)


def anreg(t):
    """Ortsvektor der anregenden Masse zum Zeitpunkt t. """
    t_max = 0.5
    delta_t = 0.2                         # Langer Anregungspuls.
    pos = np.empty(dim)
    pos[0] = A_long * np.exp(-((t - t_max) / delta_t) ** 2)
    pos[1] = A_tran * np.exp(-((t - t_max) / delta_t) ** 2)
    return pos


def federkraft(r1, r2):
    """Kraft auf die Masse am Ort r1. """
    L = np.linalg.norm(r2 - r1)
    F = D * (L - L0) * (r2 - r1) / L
    return F


def dgl(t, u):
    r, v = np.split(u, 2)
    r = r.reshape(N, dim)
    a = np.zeros((N, dim))

    # Addiere die Beschleunigung durch die jeweils linke Feder.
    for i in range(1, N):
        a[i] += federkraft(r[i], r[i-1]) / m

    # Addiere die Beschleunigung durch die jeweils rechte Feder.
    for i in range(N-1):
        a[i] += federkraft(r[i], r[i+1]) / m

    # Addiere die Beschleunigung durch die anregende Masse.
    a[0] += federkraft(r[0], anreg(t)) / m

    # Die letzte Masse soll festgehalten werden.
    a[N - 1] = 0

    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest. Alle N-1
# Teilchen ruhen in der Ruhelage.
v0 = np.zeros(N * dim)
u0 = np.concatenate((r0.reshape(-1), v0))

# Löse die Bewegungsgleichung bis zum Zeitpunkt T.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionals Array um:
#    1. Index - Teilchennummer
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 4))

# Erzeuge eine Axes für die animierte Darstellung der
# Masse-Feder-Kette.
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlim(-L, (N + 1) * L)
ax1.set_ylim(-1.2 * A_tran, 1.2 * A_tran)
ax1.set_ylabel('y [m]')
ax1.tick_params(labelbottom=False)
ax1.grid()

# Erzeuge je einen Punktplot (blau) für die beiden Wellen.
teilchen, = ax1.plot(r0[:, 0], r0[:, 1], 'ob')
teilchen0, = ax1.plot([-L], 0, 'or')

# Erzeuge ein Textfeld für die Angabe des Zeitpunkts.
text = ax1.text(0.97, 0.97, '', transform=ax1.transAxes,
                horizontalalignment='right',
                verticalalignment='top')

# Erzeuge eine zweite Axes für die animierte Darstellung der
# transversalen und longitudinalen Auslenkung.
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlim(-L, (N + 1) * L)
ax2.set_ylim(-1.2 * max(A_tran, A_long),
             1.2 * max(A_tran, A_long))
ax2.set_ylabel('u [m]')
ax2.set_xlabel('x [m]')
ax2.grid()

# Erzeuge je einen Linienplot für die Momentanauslenkung.
# Dabei wollen wir die anregende Masse mit einschließen.
xw = np.linspace(0, N * L, N + 1)
welle_trans, = ax2.plot(xw, 0 * xw, '-', label='trans')
welle_long, = ax2.plot(xw, 0*xw, '-', label='long')

# Füge eine Legende hinzu.
ax2.legend(loc='upper left')


def update(n):
    # Aktualisiere die Position der simulierten Massen.
    teilchen.set_data(r[:, :, n].T)

    # Aktualisiere die Position der anregenden Masse.
    teilchen0.set_data(anreg(t[n]))

    # Erzeuge ein Array mit den Auslenkungen aller Massen
    # aus der Ruhelage (inklusive der anregenden Masse).
    w = np.concatenate([anreg(t[n]).reshape(1, dim),
                        r[:, :, n] - r0])

    # Aktualisiere den Plot für die Transversal- und
    # Longitudinalwelle.
    welle_long.set_ydata(w[:, 0])
    welle_trans.set_ydata(w[:, 1])

    # Aktualisiere die Zeitangabe.
    text.set_text(f't = {t[n]:.2f}s')

    return teilchen, teilchen0, welle_trans, welle_long, text


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
