"""Berechnung des prozentualen Fehlers der Näherung sin(x) = x.

In diesem Beispiel wird der prozentuale Fehler der Näherung
sin(x) = x im Bereich von 5 Grad bis 90 Grad in 5
Grad-Schritten bestimmt. Das Ergebnis wird in numpy-arrays
gespeichert. """

import numpy as np

winkel = np.arange(5, 95, 5)
x = np.radians(winkel)
fehler = 100 * (x - np.sin(x)) / np.sin(x)

print(winkel)
print(fehler)
