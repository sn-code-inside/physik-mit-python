"""Funktionsgraph der Sinus- und Kosinusfunktion.

Dieses Beispiel erzeugt die Funktionsgraphen der Sinus- und
Kosinusfunktion inkl. der üblichen Beschriftungen. """

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 360, 500)
y1 = np.sin(np.radians(x))
y2 = np.cos(np.radians(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_title('Sinus- und Kosinusfunktion')
ax.set_xlabel('Winkel [Grad]')
ax.set_ylabel('Funktionswert')
ax.set_xlim(0, 360)
ax.set_ylim(-1.1, 1.1)
ax.grid()

ax.plot(x, y1, label='Sinus')
ax.plot(x, y2, label='Kosinus')
ax.legend()

plt.show()
