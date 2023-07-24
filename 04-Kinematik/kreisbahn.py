"""Animation zur Beschleunigung und Geschwindigkeit der
gleichförmigen Kreisbewegung. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Parameter der Simulation.
R = 3.0                      # Radius der Kreisbahn [m].
T = 12.0                     # Umlaufdauer [s].
dt = 0.02                    # Zeitschrittweite [s].
omega = 2 * np.pi / T        # Winkelgeschwindigkeit [1/s].

# Gib das analytische Ergebnis aus.
print(f'Bahngeschwindigkeit:       {R*omega:.3f} m/s')
print(f'Zentripetalbeschleunigung: {R*omega**2:.3f} m/s²')

# Erzeuge ein Array von Zeitpunkten für einen Umlauf.
t = np.arange(0, T, dt)

# Erzeuge ein leeres n x 2 - Arrray für die Ortsvektoren.
r = np.empty((t.size, 2))

# Berechne die Position des Massenpunktes für jeden Zeitpunkt.
r[:, 0] = R * np.cos(omega * t)
r[:, 1] = R * np.sin(omega * t)

# Berechne den Geschwindigkeits- und Beschleunigungsvektor.
v = (r[1:, :] - r[:-1, :]) / dt
a = (v[1:, :] - v[:-1, :]) / dt

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(-1.2 * R, 1.2 * R)
ax.set_ylim(-1.2 * R, 1.2 * R)
ax.set_aspect('equal')
ax.grid()

# Plotte die Kreisbahn.
plot, = ax.plot(r[:, 0], r[:, 1])

# Erzeuge einen Kreis, der die Position der Masse darstellt.
punkt, = ax.plot([0], [0], 'o', color='blue')

# Erzeuge zwei Textfelder für die Anzeige des aktuellen
# Geschwindigkeits- und Beschleunigungsbetrags.
text_v = ax.text(0, 0.2, '', color='red')
text_a = ax.text(0, -0.2, '', color='black')

# Erzeuge Pfeile für die Gewschwindigkeit und die Beschleunigung.
style = mpl.patches.ArrowStyle.Simple(head_length=6,
                                      head_width=3)
arrow_v = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='red',
                                      arrowstyle=style)
arrow_a = mpl.patches.FancyArrowPatch((0, 0), (0, 0),
                                      color='black',
                                      arrowstyle=style)

# Füge die Grafikobjekte zur Axes hinzu.
ax.add_artist(arrow_v)
ax.add_artist(arrow_a)


def update(n):
    # Aktualisiere den Geschwindigkeitspfeil und zeige den
    # Geschwindigkeitsbetrag an.
    if n < v.shape[0]:
        arrow_v.set_positions(r[n], r[n] + v[n])
        text_v.set_text(f'v = {np.linalg.norm(v[n]):.3f} m/s')

    # Aktualisiere den Beschleunigungspfeil und zeige den
    # Beschleunigungssbetrag an.
    if n < a.shape[0]:
        arrow_a.set_positions(r[n], r[n] + a[n])
        text_a.set_text(f'a = {np.linalg.norm(a[n]):.3f} m/s²')

    # Aktualisiere die Position des Punktes.
    punkt.set_data(r[n])

    return punkt, arrow_v, arrow_a, text_a, text_v


# Erzeuge das Animationsobjekt und starte die Animation.
ani = mpl.animation.FuncAnimation(fig, update, interval=30,
                                  frames=t.size, blit=True)
plt.show()
