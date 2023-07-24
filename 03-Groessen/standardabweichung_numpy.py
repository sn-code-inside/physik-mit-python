"""Berechnung des Mittelwerts, der Standardabweichung und des
Fehlers des Mittelwertes mithilfe von NumPy. """

import math
import numpy as np

# Gemessene Schwingungsdauern [s].
T = np.array([2.05, 1.99, 2.06, 1.97, 2.01,
              2.00, 2.03, 1.97, 2.02, 1.96])

# Berechne die drei gesuchten Kenngrößen.
mittel = np.mean(T)
sigma = np.std(T)
delta_T = sigma / math.sqrt(T.size)

print(f'Mittelwert:             <T> = {mittel:.2f} s')
print(f'Standardabweichung:   sigma = {sigma:.2f} s' )
print(f'Mittlerer Fehler:   Delta T = {delta_T:.2f} s')
