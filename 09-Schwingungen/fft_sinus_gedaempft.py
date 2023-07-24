"""Fourier-Transformation einer gedämpften Sinusschwingung. """

import numpy as np
import matplotlib.pyplot as plt

# Zeitdauer des Signals [s] und Abtastrate [1/s].
T = 0.2
rate = 44100

# Erzeuge ein gedämpftes, sinusförmiges Signal.
t = np.arange(0, T, 1 / rate)
x = np.sin(2 * np.pi * 500 * t) * np.exp(-30 * t)

# Führe die Fourier-Transformation durch.
x_ft = np.fft.fft(x) / x.size

# Bestimme die zugehörigen Frequenzen.
freq = np.fft.fftfreq(x.size, d=1/rate)

# Sortiere die Frequenzen in aufsteigender Reihenfolge.
freq = np.fft.fftshift(freq)
x_ft = np.fft.fftshift(x_ft)

# Erzeuge eine Figure und ein GridSpec-Objekt.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)
gs = fig.add_gridspec(2, 2)

# Erzeuge eine Axes und plotte den Zeitverlauf.
ax1 = fig.add_subplot(gs[:, 0])
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Auslenkung')
ax1.grid()
ax1.plot(t, x)

# Erzeuge eine Axes und plotte Real- und Imaginärteil der
# Fourier-Transformierten.
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_xlabel('Frequenz [Hz]')
ax2.set_ylabel('Amplitude')
ax2.set_xlim(-1000, 1000)
ax2.grid()
ax2.plot(freq, np.real(x_ft), 'r', label='Realteil')
ax2.plot(freq, np.imag(x_ft), 'b', label='Imaginärteil')
ax2.legend(loc='upper right')

# Erzeuge eine Axes und plotte den Betrag der
# Fourier-Transformierten.
ax3 = fig.add_subplot(gs[1, 1])
ax3.set_xlabel('Frequenz [Hz]')
ax3.set_ylabel('Amplitude')
ax3.set_xlim(-1000, 1000)
ax3.grid()
ax3.plot(freq, np.abs(x_ft), label='Betrag')
ax3.legend()

# Zeige den Plot an.
plt.show()
