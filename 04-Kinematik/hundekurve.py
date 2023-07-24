"""Simulation der Hundekurve. """

import numpy as np
import matplotlib.pyplot as plt

# Startposition (x, y) des Hundes [m].
r0_hund = np.array([0.0, 10.0])

# Startposition (x, y) des Menschen [m].
r0_mensch = np.array([0.0, 0.0])

# Vektor der Geschwindigkeit (vx, vy) des Menschen [m/s].
v0_mensch = np.array([2.0, 0.0])

# Betrag der Geschwindigkeit des Hundes [m/s].
v0_hund = 3.0

# Maximale Simulationsdauer [s].
t_max = 500

# Zeitschrittweite [s].
dt = 0.01

# Brich die Simulation ab, wenn der Abstand von Hund und
# Mensch kleiner als epsilon ist.
epsilon = v0_hund * dt

# Lege Listen an, um das Simulationsergebnis zu speichern.
t = [0]
r_hund = [r0_hund]
r_mensch = [r0_mensch]
v_hund = []

# Schleife der Simulation.
while True:
    # Berechne den Geschwindigkeitsvektor des Hundes.
    delta_r = r_mensch[-1] - r_hund[-1]
    v = v0_hund * delta_r / np.linalg.norm(delta_r)
    v_hund.append(v)

    # Beende die Simulation, wenn der Abstand von Hund und
    # Mensch klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (np.linalg.norm(delta_r) < epsilon) or (t[-1] > t_max):
        break

    # Berechne die neue Position von Hund und Mensch und die
    # neue Zeit.
    r_hund.append(r_hund[-1] + dt * v)
    r_mensch.append(r_mensch[-1] + dt * v0_mensch)
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r_hund = np.array(r_hund)
v_hund = np.array(v_hund)
r_mensch = np.array(r_mensch)

# Berechne den Beschleunigungsvektor des Hundes für alle
# Zeitpunkte. Achtung! Das Array a_hund hat eine Zeile weniger,
# als es Zeitpunkte gibt.
a_hund = (v_hund[1:, :] - v_hund[:-1, :]) / dt

# Erzeuge eine Figure der Größe 10 inch x 3 inch.
fig = plt.figure(figsize=(10, 3))
fig.set_tight_layout(True)

# Plotte die Bahnkurve des Hundes.
ax1 = fig.add_subplot(1, 3, 1)
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')
ax1.set_aspect('equal')
ax1.grid()
ax1.plot(r_hund[:, 0], r_hund[:, 1])

# Plotte den Abstand von Hund und Mensch.
ax2 = fig.add_subplot(1, 3, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Abstand [m]')
ax2.grid()
ax2.plot(t, np.linalg.norm(r_hund - r_mensch, axis=1))

# Plotte den Betrag der Beschleunigung des Hundes.
ax3 = fig.add_subplot(1, 3, 3)
ax3.set_xlabel('t [s]')
ax3.set_ylabel('Beschl. [m/s²]')
ax3.grid()
ax3.plot(t[1:], np.linalg.norm(a_hund, axis=1))

# Zeige die Grafik an.
plt.show()
