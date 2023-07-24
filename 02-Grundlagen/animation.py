"""Animierte Darstellung einer ebenen Welle. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Erzeuge die x-Werte von 0 bis 20 in 500 Schritten.
x = np.linspace(0, 20, 500)

# Definiere die Parameter.
omega = 1.0             # Kreisfrequenz
k = 1.0                 # Wellenzahl
delta_t = 0.02          # Zeitschrittweite

# Erzeuge die Figure und das Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_ylim(-1.2, 1.2)
ax.set_xlabel('Ort x')
ax.set_ylabel('u(x, t)')
ax.grid()

# Erzeuge einen Plot und ein leeres Textfeld.
plot, = ax.plot(x, 0 * x)
text = ax.text(0.0, 1.05, '')


def update(n):
    """Berechne die Welle zum n-ten Zeitschritt und
    aktualisiere die entsprechenden Grafikelemente. """

    # Berechne die Funktion zum Zeitpunkt t.
    t = n * delta_t
    u = np.cos(k * x - omega * t)

    # Aktualisiere den Plot.
    plot.set_ydata(u)

    # Aktualisiere den Text des Textfeldes.
    text.set_text(f't = {t:5.1f}')

    # Gib ein Tupel mit den Grafikelementen zurück, die neu
    # dargestellt werden müssen.
    return plot, text


# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Animation.
plt.show()
