"""Funktionsgraph der Sinusfunktion. """

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 360, 500)
y = np.sin(np.radians(x))

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(x, y)

plt.show()
