"""Simulation eines Swing-by-Manövers. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.integrate

# Konstanten 1 Tag [s] und 1 Jahr [s].
stunde = 60 * 60
tag = 24 * stunde
jahr = 365.25 * tag

# Simulationszeitdauer T und dargestellte Schrittweite dt [s].
T = 36 * tag
dt = 1 * stunde

# Graviationskonstante [m³ / (kg * s²)].
G = 6.674e-11

# Masse des Jupiters und der Raumsonde [kg].
m1 = 1.898e27
m2 = 1e3

# Anflugwinkel der Raumsonde relative zur Bewegungsrichtung
# des Jupiters
alpha = math.radians(60)

# Anfangsentfernung der Raumsone vom Koordinatenursprung.
d_sonde = 15e9

# Anfangsentfernung des Jupiters vom Koordinatenursprung.
d_jupiter = 20.18e9

# Radius des Jupiters.
R_jupiter = 6.9911e7

# Anfangspositionen der Körper [m].
r0_1 = np.array([d_jupiter, 0.0])
r0_2 = d_sonde * np.array([-np.cos(alpha), -np.sin(alpha)])

# Anfangsgeschwindigkeiten der Körper [m/s].
v0_1 = np.array([-13e3, 0])
v0_2 = 9e3 * np.array([np.cos(alpha), np.sin(alpha)])


def dgl(t, u):
    r1, r2, v1, v2 = np.split(u, 4)
    a1 = G * m2 / np.linalg.norm(r2 - r1)**3 * (r2 - r1)
    a2 = G * m1 / np.linalg.norm(r1 - r2)**3 * (r1 - r2)
    return np.concatenate([v1, v2, a1, a2])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0_1, r0_2, v0_1, v0_2))

# Löse die Bewegungsgleichung numerisch.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r1, r2, v1, v2 = np.split(result.y, 4)

# Berechne den Abstand der Raumsonde vom Jupiter.
abstand = np.linalg.norm(r1 - r2, axis=0)

# Gibt die minimale Entfernung der Raumsonde vom Jupiter in
# Vielfachen des Jupiterradius an.
abstand_min = np.min(abstand)
print(f"Min. Abstand: {abstand_min/R_jupiter:.1f} Jupiterradien")

# Berechne den Geschwindigkeitsbetrag der Raumsonde.
geschwindigkeit = np.linalg.norm(v2, axis=0)

# Erzeuge eine Figure und eine Axes für die Animation.
fig = plt.figure()
fig.set_tight_layout(True)
ax1 = fig.add_subplot(1, 1, 1)
ax1.set_xlabel('$x$ [m]')
ax1.set_ylabel('$y$ [m]')
ax1.set_xlabel('$x$ [m]')
ax1.grid()

# Plotte die Bahnkurve der beiden Körper.
ax1.plot(r1[0], r1[1], '-r')
ax1.plot(r2[0], r2[1], '-b')

# Erzeuge eine zweite Figure für die Plots.
fig2 = plt.figure()
fig2.set_tight_layout(True)

# Erzeuge eine Axes für den Abstand als Funktion der Zeit.
ax2 = fig2.add_subplot(1, 2, 1)
ax2.set_xlabel("$t$ [Tage]")
ax2.set_ylabel("Abstand [m]")
ax2.grid()
ax2.plot(t / tag, abstand)

# Erzeuge eine Axes für die Geschwindigkeit der Raumsonde.
ax3 = fig2.add_subplot(1, 2, 2)
ax3.set_xlabel("$t$ [Tage]")
ax3.set_ylabel("Geschwindigkeit [km/s]")
ax3.grid()
ax3.plot(t / tag, geschwindigkeit / 1e3)

# Erzeuge eine Punktplot, für die Positionen der Himmelskörper.
koerper1, = ax1.plot([0], [0], 'o', color='red')
koerper2, = ax1.plot([0], [0], 'o', color='blue')


def update(n):
    # Aktualisiere die Position des Himmelskörper.
    koerper1.set_data(r1[:, n])
    koerper2.set_data(r2[:, n])

    return koerper1, koerper2


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=40,
                                  frames=t.size, blit=True)
plt.show()
