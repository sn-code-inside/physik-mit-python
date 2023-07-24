"""Darstellung und Flächenberechnung eines Blumenbeets. """

import numpy as np
import matplotlib.pyplot as plt

# Koordinaten der Eckpunkte [m].
x = np.array([0.0, 0.0, 1.0, 2.2, 2.8, 3.8, 4.6,
              5.7, 6.4, 7.1, 7.6, 7.9, 7.9, 0.0])
y = np.array([1.0, 2.8, 3.3, 3.5, 3.4, 2.7, 2.4,
              2.3, 2.1, 1.6, 0.9, 0.5, 0.0, 1.0])

# Berechnung der Fläche mit der Gaußschen Trapezformel.
A = 0.5 * abs((y + np.roll(y, 1)) @ (x - np.roll(x, 1)))

# Ausgabe als Antwortsatz.
print(f"Die Fläche beträgt {A:.1f} m².")

# Erzeuge die Figure und das Axes-Objekt. Damit das Blumenbeet
# nicht verzerrt dargestellt wird, sorgen wir mit
#    ax.set_aspect('equal')
# dafür, dass die Skalierung in x- und y-Richtung gleich ist.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
ax.grid()

# Erzeuge einen Plot und ein leeres Textfeld.
ax.plot(x, y)

# Wähle einen Punkt, ungefähr in der Mitte des Beetes und
# erzeuge dort ein Textfeld mit der Flächenangabe.
xm = 0.5 * (np.min(x) + np.max(x))
ym = 0.5 * (np.min(y) + np.max(y))
ax.text(xm, ym, f'A = {A:.1f} m²')

# Zeige den Plot.
plt.show()
