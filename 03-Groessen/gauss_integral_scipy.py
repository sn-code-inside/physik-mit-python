"""Numerische Integration der Gauß-Verteilung mit SciPy. """

import math
import scipy.integrate

sigma = 0.5       # Standardabweichung.
x_max = 3         # Integrationsbereich von -x_max bis +x_max.


def f(x):
    """Gauß-Verteilung, Standardabw. sigma, Mittelwert 0. """
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    return a * math.exp(- x**2 / (2 * sigma**2))


p, err = scipy.integrate.quad(f, -x_max, x_max)

print(f'Ergebnis der Integration: {p}')
print(f'Fehler der Integration:   {err}')
