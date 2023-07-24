"""Amplitudenmodulierte Sinusschwingung. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import sounddevice

# Simulationsdauer [s].
T = 3.0

# Im Plot dargestellter Zeitbereich [s].
T_plot = [0, 0.1]

# Abtastrate [1/s].
rate = 44100

# Kreisfrequenz der Trägerschwingung.
omega_0 = 2 * np.pi * 400

# Kreisfrequenz der Modulation [s].
omega_mod = 2 * np.pi * 20


def s(t):
    """Signal als Funktion der Zeit. """
    return np.sin(omega_mod * t)


def y(t):
    """Amplitudenmodulierter Träger. """
    return 0.5 * (1 + s(t)) * np.sin(omega_0 * t)


# Erzeuge eine Zeitachse.
t = np.arange(0, T, 1/rate)

# Erzeuge eine Figure ein Gridspec-Objekt.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)
gs = fig.add_gridspec(2, 2)

# Plotte das zu modulierende Signal.
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_ylabel('s(t)')
ax1.tick_params(labelbottom=False)
ax1.set_xlim(T_plot)
ax1.grid()
ax1.plot(t, s(t))

# Plotte den amplitudenmodulierten Träger.
ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
ax2.set_ylabel('y(t)')
ax2.set_xlabel('t [s]')
ax2.grid()
ax2.plot(t, y(t))

# Führe die Fourier-Transformation durch.
y_ft = np.fft.fft(y(t)) / t.size
freq = np.fft.fftfreq(t.size, d=1/rate)
freq = np.fft.fftshift(freq)
y_ft = np.fft.fftshift(y_ft)

# Plotte das Spektrum des amplitudenmodulierten Trägers.
ax3 = fig.add_subplot(gs[:, 1])
ax3.grid()
ax3.set_xlabel('f [Hz]')
ax3.set_ylabel('Amplitude')
ax3.set_xlim(400 - 40, 400 + 40)
ax3.plot(freq, np.abs(y_ft), '.-')

# Gib den amplitudenmodulierten Träger als Audiodatei aus.
dat = np.int16(y(t) / np.max(np.abs(y(t))) * 32767)
scipy.io.wavfile.write('output.wav', rate, dat)

# Gib das Signal als Sound aus.
sounddevice.play(dat, rate, blocking=True)

# Zeige den Plot an.
plt.show()
