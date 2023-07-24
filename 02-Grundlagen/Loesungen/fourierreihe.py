"""Animation der Fourier-Reihe der Rechteckfunktion. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Anzahl der Punkte, die dargestellt werden sollen.
N = 2000

# Erzeuge ein Array mit den x-Werten.
x = np.linspace(0, 2 * np.pi, N)

# Erzeuge die Figure und das Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Fourier-Approximation einer Rechteckfunktion')
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid()

# Erzeuge einen Plot und ein leeres Textfeld.
plot, = ax.plot(x, 0 * x)
text = ax.text(np.pi / 2, 0.2, '')


def update(n):
    """Berechne die ersten n Glieder der Fourierreihe und
    aktualisiere die entsprechenden Grafikelemente. """

    # Berechne die ersten n Summanden der Fourier-Reihe.
    y = np.zeros(N)
    for k in range(n):
        y += 4 / np.pi * np.sin((2*k+1) * x) / (2 * k + 1)

    # Aktualisiere die y-Werte des Plots.
    plot.set_ydata(y)

    # Aktualisiere den Text des Textfeldes.
    text.set_text(f'n = {n}')

    # Gib ein Tupel mit den Grafikelementen zurück, die neu
    # dargestellt werden müssen.
    return plot, text


# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Animation.
plt.show()
