"""Simulation und animierte Darstellung eines
Verfolgungsproblems: Der Verfolger (Hund) läuft immer direkt
auf den Menschen zu, der sich auf einer kreisförmigen Bahn
bewegt. """

import math
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Startposition (x, y) des Hundes [m].
r0_hund = np.array([28.0, 0.0])

# Geschwindigkeit des Hundes [m/s].
v0_hund = 2

# Radius der Kreisbahn, auf der sich der Mensch bewegt [m].
R = 5.0

# Bahngeschwindigkeit des Menschen [m/s].
v_mensch = 2.5

# Maximale Simulationsdauer [s].
t_max = 40

# Zeitschrittweite [s].
dt = 0.01

# Breche die Simulation ab, wenn der Abstand von Hund und
# Mensch kleiner als epsilon ist.
epsilon = v0_hund * dt

# Wir legen Listen an, um die Simulationsergebnisse zu speichern.
t = [0]
r_hund = [r0_hund]
r_mensch = []
v_hund = []

# Schleife der Simulation
while True:
    # Berechne die aktuelle Position des Menschen und hänge
    # sie an die entsprechende Liste an.
    r_mensch.append(np.array(
        [R * math.cos(v_mensch / R * t[-1]),
         R * math.sin(v_mensch / R * t[-1])]))

    # Berechne den neuen Geschwindigkeitsvektor des Hundes.
    delta_r = r_mensch[-1] - r_hund[-1]
    v = v0_hund * delta_r / np.linalg.norm(delta_r)
    v_hund.append(v)

    # Beende die Simulation, wenn der Abstand von Hund und
    # Mensch klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (np.linalg.norm(delta_r) < epsilon) or (t[-1] > t_max):
        break

    # Berechne die neue Position des Hundes.
    r_hund.append(r_hund[-1] + dt * v)
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r_hund = np.array(r_hund)
v_hund = np.array(v_hund)
r_mensch = np.array(r_mensch)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-6, 30)
ax.set_ylim(-6, 6)
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen leeren Plot für die Bahnkurve des Hundes.
plot_hund, = ax.plot([], [], color='b')
plot_mensch, = ax.plot([], [], color='r')

# Erzeuge zwei Punktplots mit jeweils nur einem Punkt.
Hund, = ax.plot([0], [0], 'o', color='blue')
Mensch, = ax.plot([0], [0], 'o', color='red')

# Erzeuge einen Pfeil für die Geschwindigkeit und füge
# diesen zur Axes hinzu.
style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
ax.add_artist(arrow_v)


def update(n):
    # Lege den Anfangs- und den Zielpunkt des
    # Geschwindigkeitspfeiles fest.
    arrow_v.set_positions(r_hund[n], r_hund[n] + v_hund[n])

    # Aktualisiere die Positionen von Hund und Mensch
    Hund.set_data(r_hund[n])
    Mensch.set_data(r_mensch[n])

    # Plotte die Bahnkurve des Hundes bis zum aktuellen
    # Zeitpunkt.
    plot_hund.set_data(r_hund[:n+1, 0], r_hund[:n+1, 1])
    plot_mensch.set_data(r_mensch[:n+1, 0], r_mensch[:n+1, 1])

    return Hund, Mensch, arrow_v, plot_hund, plot_mensch


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
