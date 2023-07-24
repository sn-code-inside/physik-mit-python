"""Animierte Darstellung des Sonnensystems.

Das Programm liest die Daten des Programms sonnensystem_sim.py
aus der Datei ephemeriden.npz ein und stellt das Sonnensystem
animiert dar. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import mpl_toolkits.mplot3d

# Lies die Simulationsdaten ein.
dat = np.load('ephemeriden.npz')
tag, jahr, AE, G = dat['tag'], dat['jahr'], dat['AE'], dat['G']
T, dt = dat['T'], dat['dt']
name = dat['name']
m, t, r, v = dat['m'], dat['t'], dat['r'],  dat['v']

# Farben für die Darstellung der Planetenbahnen.
farbe = ['gold', 'darkcyan', 'orange', 'blue', 'red', 'brown',
         'olive', 'green', 'slateblue', 'black', 'gray']

# Anzahl der Himmelskörper.
N = len(name)

# Berechne die verschiedenen Energiebeiträge.
E_kin = 1/2 * m @ np.sum(v * v, axis=1)
E_pot = np.zeros(t.size)
for i in range(N):
    for j in range(i):
        dr = np.linalg.norm(r[i] - r[j], axis=0)
        E_pot -= G * m[i] * m[j] / dr
E = E_pot + E_kin

# Berechne den Gesamtimpuls.
p = m @ v.swapaxes(0, 1)

# Berechne die Position des Schwerpunktes.
rs = m @ r.swapaxes(0, 1) / np.sum(m)

# Berechne den Drehimpuls.
L = m @ np.cross(r, v, axis=1).swapaxes(0, 1)

# Erzeuge eine Figure für die Plots der Erhaltungsgrößen.
fig1 = plt.figure()
fig1.set_tight_layout(True)

# Erzeuge eine Axes und plotte die Energie.
ax1 = fig1.add_subplot(2, 2, 1)
ax1.set_title('Energie')
ax1.set_xlabel('t [d]')
ax1.set_ylabel('E [J]')
ax1.grid()
ax1.plot(t / tag, E, label='$E$')

# Erzeuge eine Axis und plotte den Impuls.
ax2 = fig1.add_subplot(2, 2, 2)
ax2.set_title('Impuls')
ax2.set_xlabel('t [d]')
ax2.set_ylabel('$\\vec p$ [kg m / s]')
ax2.grid()
ax2.plot(t / tag, p[0, :], '-r', label='$p_x$')
ax2.plot(t / tag, p[1, :], '-b', label='$p_y$')
ax2.plot(t / tag, p[2, :], '-k', label='$p_z$')
ax2.legend()

# Erzeuge eine Axis und plotte den Drehimpuls.
ax3 = fig1.add_subplot(2, 2, 3)
ax3.set_title('Drehimpuls')
ax3.set_xlabel('t [d]')
ax3.set_ylabel('$\\vec L$ [kg m² / s]')
ax3.grid()
ax3.plot(t / tag, L[0, :], '-r', label='$L_x$')
ax3.plot(t / tag, L[1, :], '-b', label='$L_y$')
ax3.plot(t / tag, L[2, :], '-k', label='$L_z$')
ax3.legend()

# Exzeuge eine Axes und plotte die Schwerpunktskoordinaten.
ax4 = fig1.add_subplot(2, 2, 4)
ax4.set_title('Schwerpunkt')
ax4.set_xlabel('t [d]')
ax4.set_ylabel('$\\vec r_s$ [m]')
ax4.grid()
ax4.plot(t / tag, rs[0, :], '-r', label='$r_{s,x}$')
ax4.plot(t / tag, rs[1, :], '-b', label='$r_{s,y}$')
ax4.plot(t / tag, rs[2, :], '-k', label='$r_{s,z}$')
ax4.legend()

# Erzeuge eine Figure und eine 3D-Axes für die Animation.
fig2 = plt.figure(figsize=(9, 6))
ax = fig2.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('x [AE]')
ax.set_ylabel('y [AE]')
ax.set_zlabel('z [AE]')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.grid()

# Plotte für jeden Planeten die Bahnkurve und füge die
# Legende hinzu.
for i in range(N):
    ax.plot(r[i, 0] / AE, r[i, 1] / AE, r[i, 2] / AE,
            '-', color=farbe[i], label=name[i])
ax.legend()

# Erzeuge für jeden Planeten einen Punktplot in der
# entsprechenden Farbe und speichere diesen in der Liste planet.
planet = []
for i in range(N):
    p, = ax.plot([0], [0], 'o', color=farbe[i])
    planet.append(p)

# Füge ein Textfeld für die Anzeige der verstrichenen Zeit hinzu.
text = fig2.text(0.5, 0.95, '')


def update(n):
    for i in range(N):
        planet[i].set_data(np.array([r[i, 0, n] / AE]),
                           np.array([r[i, 1, n] / AE]))
        planet[i].set_3d_properties(r[i, 2, n] / AE)
    text.set_text(f'{t[n] / jahr:.2f} Jahre')
    return planet + [text]


# Zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig2, update, interval=30,
                                  frames=t.size)
plt.show()
