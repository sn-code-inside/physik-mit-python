"""Numerische Integration der Gauß-Verteilung. """

import math

sigma = 0.5       # Standardabweichung.
x_max = 3         # Integrationsbereich von -x_max bis +x_max.
dx = 0.01         # Schrittweite für die Integration.


def f(x):
    """Gauß-Verteilung, Standardabw. sigma, Mittelwert 0. """
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    return a * math.exp(- x**2 / (2 * sigma**2))


x = -x_max
p = 0
while x < x_max:
    p += f(x + dx/2) * dx
    x += dx

print(p)
