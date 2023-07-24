"""Vergleich einer Schwebung mit einer Amplitudenmodulation. """

import numpy as np
import matplotlib.pyplot as plt

# Simulationsdauer [s].
T = 0.2

# Abtastrate [1/s].
rate = 44100

# Kreisfrequenzen der beiden Schwingungen [1/s].
omega1 = 2 * np.pi * 395
omega2 = 2 * np.pi * 405

# Erzeuge eine Zeitachse.
t = np.linspace(0, T, 1000)

# Erzeuge die beiden Signale.
y1 = 0.5 * np.sin(omega1 * t)
y2 = 0.5 * np.sin(omega2 * t)

# Bilde das Summensignal.
y_schweb = y1 + y2

# Berechne die passende Modulations- und Mittenkreisfrequenz
# für die Amplitudenmodulation.
omega_mod = omega2 - omega1
omega_0 = (omega1 + omega2) / 2

# Berechne amplitudenmodulierte Signal.
y_am = 0.5 * (1 + np.cos(omega_mod * t)) * np.sin(omega_0 * t)

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Stelle die Schwebung dar.
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_title('Schwebung')
ax1.set_xlabel('t [1/s]')
ax1.set_ylabel('y [a.u.]')
ax1.set_xlim(0, T)
ax1.grid()
ax1.plot(t, y_schweb)

# Stelle die Amplitudenmodulation dar.
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_title('Amplitudenmodulation')
ax2.set_xlabel('t [1/s]')
ax2.set_ylabel('y [a.u.]')
ax2.set_xlim(0, T)
ax2.grid()
ax2.plot(t, y_am)

# Zeige den Plot an.
plt.show()
