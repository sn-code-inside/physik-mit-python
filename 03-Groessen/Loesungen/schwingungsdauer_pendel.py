"""Berechnung des Wahrscheinlichkeit, dass die Messwerte der
Schwingungdauer eines Pendels innerhalb eines bestimmten
Intervalls T_min bis T_max liegen. """

import math
import numpy as np
import scipy.integrate

# Gemessene Schwingungsdauern [s].
T = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
              2.00, 2.03, 1.97, 2.02, 1.96])

# Vorgegebene Grenzen des Intervalls [s].
T_min = 1.95
T_max = 2.05

# Berechnung den Mittelwert und die Standardabweichung.
mittel = np.mean(T)
sigma = np.std(T, ddof=1)


def f(x):
    """Gauß-Verteilung der Werte. """
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    return a * math.exp(- (x - mittel)**2 / (2 * sigma**2))


# Integriere die Gauß-Verteilung über das gegebene Intervall.
p, err = scipy.integrate.quad(f, T_min, T_max)

# Gib das Ergebnis aus.
print(f'Im Intervall von {T_min:.2f} s bis {T_max:.2f} s liegen '
      f'{100*p:.1f}% der Messwerte.')

