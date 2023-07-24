"""Fourier-Transformation einer Sinusschwingung. """

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer des Signals [s] und Abtastrate [1/s].
T = 0.2
rate = 44100

# Erzeuge ein sinusförmiges Signal mit Frequenz 500 Hz.
t = np.arange(0, T, 1 / rate)
x = np.sin(2 * np.pi * 500 * t)

# Führe die Fourier-Transformation durch.
x_ft = np.fft.fft(x) / x.size

# Bestimme die zugehörigen Frequenzen.
freq = np.fft.fftfreq(x.size, d=1/rate)

# Sortiere die Frequenzen in aufsteigender Reihenfolge.
freq = np.fft.fftshift(freq)
x_ft = np.fft.fftshift(x_ft)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax2 = fig.add_subplot(1, 1, 1)
ax2.set_xlabel('Frequenz [Hz]')
ax2.set_ylabel('Amplitude')
ax2.set_xlim(-1000, 1000)
ax2.grid()

# Plotte die Fourier-Transformierte.
ax2.plot(freq, np.imag(x_ft), 'ro-', label='Imaginärteil')
ax2.plot(freq, np.real(x_ft), 'b.-', label='Realteil')
ax2.legend()

# Zeige den Plot an.
plt.show()
