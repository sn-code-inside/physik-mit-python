"""Ein einfacher Sprektrum-Analysator. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sounddevice

# Länge des Zeitfensters [s] und Abtastrate [1/s].
T = 0.5
rate = 44100

# Erzeuge eine passende Zeitachse.
t = np.arange(0, T, 1/rate)

# Bestimme die Frequenzen der Fourier-Transformation.
freq = np.fft.fftfreq(t.size, d=1/rate)
freq = np.fft.fftshift(freq)

# Plot für das Zeitsignal.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)

# Plotte das Zeitsignal.
ax1 = fig.add_subplot(2, 1, 1)
ax1.set_ylim(-1.0, 1.0)
ax1.set_xlim(0, T)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Zeitsignal [a.u.]')
ax1.grid()
plot_zeit, = ax1.plot(t, 0 * t)

# Plotte das Frequenzspektrum.
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_xlim(100, 10000)
ax2.set_ylim(0, 0.03)
ax2.set_xlabel('f [Hz]')
ax2.set_ylabel('Amplitude [a.u.]')
ax2.set_xscale('log')
ax2.grid(True, 'both')
plot_freq, = ax2.plot(freq, 0 * freq)

# Erzeuge ein Array, das das aufgenommene Zeitsignal speichert.
data = np.zeros(t.size)


def audio_callback(indata, frames, time, status):
    """Neue Audiodaten müssen verarbeitet werden. """
    global data

    # Gib im Fehlerfall eine Fehlermeldung aus.
    if status:
        print(status)

    # Kopiere die Audiodaten in das Array data.
    if frames < data.size:
        data[:] = np.roll(data, -frames)
        data[-frames:] = indata[:, 0]
    else:
        data[:] = indata[-data.size:, 0]


def update(frame):
    # Aktualisiere den Plot des Zeitsignals.
    plot_zeit.set_ydata(data)

    # Aktualisiere die Fourier-Transformierte.
    ft = np.fft.fft(data) / data.size
    ft = np.fft.fftshift(ft)
    plot_freq.set_ydata(np.abs(ft))

    return plot_zeit, plot_freq


# Erzeuge das Animationsobjekt.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Audioaufnahme und zeige die Animation an.
with sounddevice.InputStream(rate, channels=1,
                             callback=audio_callback):
    plt.show(block=True)
    
