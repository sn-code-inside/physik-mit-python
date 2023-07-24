"""Simulation und animierte Darstellung eines Bootes, das in
einer Strömung ein Ziel ansteuert. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Startposition (x, y) des Bootes [m].
r0 = np.array([-10, 2.0])

# Position (x, y) des Zieles [m].
r_ziel = np.array([0.0, 0.0])

# Vektor der Geschwindigkeit (vx, vy) der Strömung [m/s].
v_stroem = np.array([0.0, -2.5])

# Betrag der Relativgeschwindigkeit des Bootes zum Wasser [m/s].
v0_boot = 3.0

# Maximale Simulationsdauer [s].
t_max = 500

# Zeitschrittweite [s].
dt = 0.01

# Brich die Simulation ab, wenn der Abstand von Boot und
# Ziel kleiner als epsilon ist.
epsilon = v0_boot * dt

# Legen Listen an, um die Simulationsergebnisse zu speichern.
t = [0]
r = [r0]     # Ortsvektor des Bootes.
v = []       # Geschwindigkeitsvektor des Bootes.
d = []       # Richtungsvektor (Einheitsvektor) der Bootsspitze.

# Schleife der Simulation.
while True:
    # Lege die Richtung fest, in die das Boot steuert.
    delta_r = r_ziel - r[-1]
    dir_boot = delta_r / np.linalg.norm(delta_r)
    d.append(dir_boot)

    # Berechne den neuen Geschwindigkeitsvektor des Bootes.
    v.append(v0_boot * dir_boot + v_stroem)

    # Beende die Simulation, wenn der Abstand von Boot und
    # Ziel klein genug ist oder die maximale Simulationszeit
    # überschritten ist.
    if (np.linalg.norm(delta_r) < epsilon) or (t[-1] > t_max):
        break

    # Berechne die neue Position des Bootes und die neue Zeit.
    r.append(r[-1] + dt * v[-1])
    t.append(t[-1] + dt)

# Wandele die Listen in Arrays um. Die Zeilen entsprechen den
# Zeitpunkten und die Spalten entsprechen den Koordinaten.
t = np.array(t)
r = np.array(r)
v = np.array(v)
d = np.array(d)

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-11, 1)
ax.set_ylim(-5, 3)
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen leeren Plot für die Bahnkurve des Bootes
plot, = ax.plot([], [])

# Erzeuge zwei Punktplots, mit jeweils nur einem Punkt.
boot, = ax.plot([0], [0], 'o', color='blue')
ziel, = ax.plot(r_ziel[0], r_ziel[1], 'o', color='red')

# Erzeuge zwei Pfeile für die Geschwindigkeit und die aktuelle
# Ausrichtung des Bootes
style = mpl.patches.ArrowStyle.Simple(head_length=6, head_width=3)
arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
arrow_d = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_artist(arrow_v)
ax.add_artist(arrow_d)


def update(n):
    # Lege den Anfangs- und den Zielpunkt des
    # Geschwindigkeitspfeiles fest.
    arrow_v.set_positions(r[n], r[n] + v[n])

    # Lege den Anfangs und den Zielpunkt des
    # Pfeiles für die Bootsausrichtung fest.
    arrow_d.set_positions(r[n], r[n] + d[n])

    # Aktualisiere die Position des Bootes.
    boot.set_data(r[n])

    # Plotte die Bahnkurve des Bootes bis zum aktuellen
    # Zeitpunkt.
    plot.set_data(r[:n+1, 0], r[:n+1, 1])

    return boot, arrow_v, arrow_d, plot


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
