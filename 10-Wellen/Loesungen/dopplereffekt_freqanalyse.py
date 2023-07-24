"""Doppler-Effekt: Synchrone Audio- und Videoausgabe mit
Darstellung des emfangenen Frequenzspektrums. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sounddevice

# Simulationsdauer [s] und Abtastrate [1/s].
T = 10.0
rate = 44100

# Zeitdauer für das Fenster der Fourier-Transformation [s].
T_fft = 1.0

# Anzahl der Datenpunkte in der Fourier-Transformierten.
N_fft = int(T_fft * rate)

# Frequenz, mit der die Quelle Schallwellen aussendet [Hz].
f_Q = 300.0

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 340.0

# Dargestellter Koordinatenbereich [m].
xlim = (-160, 160)
ylim = (-30, 30)

# Lege die Startposition der Quelle und des Beobachters fest [m].
r0_Q = np.array([-150.0, 5.0])
r0_B = np.array([0.0, -5.0])

# Lege die Geschwindigkeit von Quelle und Beobachter fest [m/s].
v_Q = np.array([30.0, 0.0])
v_B = np.array([0.0, 0.0])


def signal(t):
    """Ausgesendetes Signal als Funktion der Zeit. """
    sig = np.sin(2 * np.pi * f_Q * t)
    sig[t < 0] = 0.0
    return sig


# Erzeuge ein Array von Zeitpunkten und lege ein leeres Array
# für das empfangene Signal an.
t = np.arange(0, T, 1/rate)
y = np.zeros(t.size)

# Berechne für jeden Zeitpunkt die beiden Positionen.
r_Q = r0_Q + v_Q * t.reshape(-1, 1)
r_B = r0_B + v_B * t.reshape(-1, 1)

# Berechne für jeden Zeitpunkt t, zu dem der Beobachter ein
# Signal auffängt, die Zeitverzögerung dt, die das Signal
# von der Quelle benötigt hat. Dazu ist eine quadratische
# Gleichung der Form
#             dt²  - 2 a dt - b = 0
# mit den unten definierten Größen a und b zu lösen.
r = r_B - r_Q
a = np.sum(v_Q * r, axis=1) / (c ** 2 - v_Q @ v_Q)
b = np.sum(r ** 2, axis=1) / (c ** 2 - v_Q @ v_Q)

# Berechne die beiden Lösungen der quadratischen Gleichung.
dt1 = a + np.sqrt(a ** 2 + b)
dt2 = a - np.sqrt(a ** 2 + b)

# Berücksichtige das Signal der positiven Lösungen.
# Beachte, dass die Amplitude mit 1/r abfällt.
ind = dt1 > 0
y[ind] = signal(t[ind] - dt1[ind]) / (c * dt1[ind])

ind = dt2 > 0
y[ind] += signal(t[ind] - dt2[ind]) / (c * dt2[ind])

# Normiere das Signal auf den Wertebereich -1 ... +1.
if np.max(np.abs(y)) > 0:
    y = y / np.max(np.abs(y))

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 6))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Animation von Quelle und Beobachter.
ax1 = fig.add_subplot(2, 1, 1)
ax1.grid()
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_aspect('equal')
ax1.set_xlabel('x [m]')
ax1.set_ylabel('y [m]')

# Erzeuge eine Axes für das Frequenzspektrum.
ax2 = fig.add_subplot(2, 1, 2)
ax2.grid()
ax2.set_xlim(260, 340)
ax2.set_ylim(0, 1)
ax2.set_xlabel('f [Hz]')
ax2.set_ylabel('Amplitude [a.u.]')

# Erzeuge zwei Kreise für Sender und Empfänger.
sender, = ax1.plot([0], [0], 'o', color='black')
empf, = ax1.plot([0], [0], 'o', color='blue')

# Erzeuge die Frequenzachse für die Fourier-Transformation.
freq = np.fft.fftfreq(N_fft, d=1/rate)
freq = np.fft.fftshift(freq)

# Erzeuge eine Linienplot für die Fourier-Transformation.
plot_fft, = ax2.plot(freq, 0 * freq)


# Startindex für die nächste Audioausgabe.
audio_index = 0


def audio_callback(outdata, frames, time, status):
    """Neue Audiodaten müssen bereitgestellt werden. """
    global audio_index

    # Gib im Fehlerfall eine Fehlermeldung aus.
    if status:
        print(status)

    # Extrahiere den benötigten Ausschnitt aus den Daten.
    # Durch das Slicing kann es passieren, dass 'dat' weniger
    # Datenpunkte als die Anzahl der 'frames' enthält.
    dat = y[audio_index: audio_index + frames]

    # Schreibe die Daten in das Ausgabe-Array.
    outdata[:dat.size, 0] = dat

    # Fülle das Ausgabe-Array ggf. mit Nullen auf.
    if dat.size < frames:
        outdata[dat.size:, 0] = 0.0

    # Erhöhe den Index um die Anzahl der verwendeten Datenpunkte.
    audio_index += dat.size


def update(n):
    """Aktualisiere die Grafik. """

    # Am Ende sollen der Sender und Empfänger stehen bleiben.
    # Ansonsten sollen die Positionen von Sender und Empfänger
    # synchron zu dem gerade ausgegebenen Audiosignal sein.
    n = min(audio_index, t.size - 1)

    # Aktualisiere die Positionen von Sender und Empfänger.
    sender.set_data(r_Q[n])
    empf.set_data(r_B[n])

    # Aktualisiere die Fourier-Transformation.
    if n > N_fft:
        ft = np.fft.fft(y[n - N_fft:n]) / N_fft
        ft = np.fft.fftshift(ft)
        ft = np.abs(ft)
        ft /= np.max(ft)
        plot_fft.set_ydata(ft)

    return sender, empf, plot_fft


# Erzeuge einen Ausgabestrom für die Audioausgabe.
stream = sounddevice.OutputStream(rate, channels=1,
                                  callback=audio_callback)

# Erzeuge die Animation.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Audioausgabe und die Animation.
with stream:
    plt.show(block=True)
