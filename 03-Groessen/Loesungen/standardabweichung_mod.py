"""Berechnung des Mittelwerts, der Standardabweichung und des
mittleren Fehlers des Mittelwertes.

Modifzierte Version, die viele Dezimalstellen ausgibt. """

import math
import numpy as np

# Gemessene Schwingungsdauern [s].
T = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
              2.00, 2.03, 1.97, 2.02, 1.96])

# Anzahl der Messwerte.
n = T.size

# Berechne den Mittelwert.
mittel = 0
for x in T:
    mittel += x
mittel /= n

# Berechne die Standardabweichung.
sigma = 0
for x in T:
    sigma += (x - mittel) ** 2
sigma = math.sqrt(sigma / (n - 1))

# Berechne den mittleren Fehler des Mittelwertes.
delta_T = sigma / math.sqrt(n)

print(f'Mittelwert:             <T> = {mittel:.6f} s')
print(f'Standardabweichung:   sigma = {sigma:.6f} s' )
print(f'Mittlerer Fehler:   Delta T = {delta_T:.6f} s')

