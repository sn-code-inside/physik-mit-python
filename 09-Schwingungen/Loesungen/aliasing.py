"""Veranschaulichung des Aliasing-Effekts bei der Abtastung
eines Signals. """

import numpy as np
import matplotlib.pyplot as plt

# Darstellte Zeitdauer [s].
T = 20

# Abtastintervall [s].
dt = 1.0

# Fein aufgelöstes Zeitraster.
t = np.linspace(0, T, 1000)

# Zeitraster mit dem Abtastintervall dt.
t1 = np.arange(0, T + dt, dt)


# Betrachtetes Signal.
def signal(freq, t):
    return np.sin(2 * np.pi * freq * t - np.pi / 4)


# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Erzeuge eine Axes für das erste Signal.
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Auslenkung')
ax1.set_xlim(0, T)
ax1.grid()

# Plotte die Abtastung eines Signals der Frequenz 1 Hz
# mit der Abtrastrate 1 Hz.
ax1.plot(t, signal(1.0, t), 'b', label='f=1 Hz')
ax1.plot(t, signal(0.0, t), 'k', label='f=0 Hz')
ax1.plot(t1, signal(0.0, t1), 'or', label='Abtastung')
ax1.legend(loc='upper right')

# Erzeuge eine Axes für das zweite Signal.
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Auslenkung')
ax2.set_xlim(0, T)
ax2.grid()

# Plotte die Abtastung eines Signals der Frequenz 1.05 Hz
# mit der Abtrastrate 1 Hz.
ax2.plot(t, signal(1.05, t), 'b', label='f=1,05 Hz')
ax2.plot(t, signal(0.05, t), 'k', label='f=0,05 Hz')
ax2.plot(t1, signal(1.05, t1), 'or', label='Abtastung')
ax2.legend(loc='upper right')

# Zeige die Grafik an.
plt.show()
