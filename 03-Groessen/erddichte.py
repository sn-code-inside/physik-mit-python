"""Berechnung der mittleren Dichte der Erde. """

import math

R = 6371e3                       # Mittlerer Erdradius [m].
m = 5.972e24                     # Masse der Erde [kg].

V = 4 / 3 * math.pi * R**3       # Berechnung des Volumens.
rho = m / V                      # Berechnung der Dichte.

print(f'Die mittlere Erddichte beträgt {rho/1e3:.3f} g / cm³.')

