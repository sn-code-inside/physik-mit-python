"""Berechnung des prozentualen Fehlers der Näherung sin(x) = x.

In diesem Beispiel wird der prozentuale Fehler der Näherung
sin(x) = x im Bereich von 5 Grad bis 90 Grad in 5
Grad-Schritten bestimmt. Das Ergebnis wird in zwei Listen
gespeichert. """

import math

liste_winkel = list(range(5, 95, 5))     # Winkel in Grad
liste_fehler = []                        # Relative Fehler in %
for winkel in liste_winkel:
    x = math.radians(winkel)
    fehler = 100 * (x - math.sin(x)) / math.sin(x)
    liste_fehler.append(fehler)

print(liste_winkel)
print(liste_fehler)
