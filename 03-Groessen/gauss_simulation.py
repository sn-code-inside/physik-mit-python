"""Simulation der Gauß-Verteilung von Messwerten. """

import numpy as np
import math
import matplotlib.pyplot as plt

n_mess = 50000                     # Anzahl der Messwerte.
n_stoer = 20                       # Anzahl der Störgrößen.

# Erzeuge die simulierten Messwerte.
mess = np.random.rand(n_mess, n_stoer)
mess = np.sum(mess, axis=1)

# Bestimme den Mittelwert und die Standardabweichung.
mittel = np.mean(mess)
sigma = np.std(mess, ddof=1)


def f(x):
    """Erwartete Gauß-Verteilung der simulierten Messwerte. """
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    return a * np.exp(- (x - mittel)**2 / (2 * sigma**2))


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Messwert')
ax.set_ylabel('Wahrscheinlichkeitsdichte')
ax.grid()

# Erzeuge ein Histogramm der simulierten Messwerte.
p, bins, patches = ax.hist(mess, bins=51, density=True)

# Werte die Gauß-Verteilung an den Rändern der einzelnen
# Histogrammbalken aus und plotte sie.
ax.plot(bins, f(bins))

# Zeige die Grafik an.
plt.show()
