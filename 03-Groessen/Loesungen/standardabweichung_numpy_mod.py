"""Berechnung des Mittelwerts, der Standardabweichung und des
mittleren Fehlers des Mittelwertes mithilfe von numpy.

Modifzierte Version, die viele Dezimalstellen ausgibt. """

import math
import numpy as np

# Gemessene Schwingungsdauern [s].
T = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
              2.00, 2.03, 1.97, 2.02, 1.96])

# Berechne die drei gesuchten Kenngrößen.
mittel = np.mean(T)
sigma = np.std(T)
delta_T = sigma / math.sqrt(T.size)

print(f'Mittelwert:             <T> = {mittel:.6f} s')
print(f'Standardabweichung:   sigma = {sigma:.6f} s' )
print(f'Mittlerer Fehler:   Delta T = {delta_T:.6f} s')
