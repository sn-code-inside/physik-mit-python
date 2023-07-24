"""Abstand des Kometen 9P/Tempel 1 von der Erde bzw. von der
Sonne als  Funktion der Zeit. """

import numpy as np
import matplotlib.pyplot as plt

# Lies die Simulationsdaten ein.
dat = np.load('ephemeriden.npz')
tag, jahr, AE, G = dat['tag'], dat['jahr'], dat['AE'], dat['G']
T, dt = dat['T'], dat['dt']
m, name = dat['m'], dat['name']
t, r, v = dat['t'], dat['r'],  dat['v']

# Berechne den Abstand zwischen Tempel1 und Erde sowie zwischen
# Tempel1 und Sonne.
d_erde = np.linalg.norm(r[9] - r[3], axis=0)
d_sonne = np.linalg.norm(r[9] - r[0], axis=0)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('t [Tage]')
ax.set_ylabel('d [AE]')
ax.grid()

# Plotte die Abstände und erzeuge eine Legende.
ax.plot(t / tag, d_erde / AE, label='Tempel1 - Erde')
ax.plot(t / tag, d_sonne / AE, label='Tempel1 - Sonne')
ax.legend()

# Zeige die Grafik an.
plt.show()
