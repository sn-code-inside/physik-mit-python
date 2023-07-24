"""Doppler-Effekt. Erzeugen einer Videodatei mit Audiospur. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import scipy.io.wavfile
import os

# Simulationsdauer [s] und Abtastrate [1/s].
T = 10.0
rate = 44100

# Bildrate (frames per second), für das Video [1/s].
fps = 30

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

# Lege ein Array für das empfangende Signal an.
y = np.zeros(t.size)

# Berechne die Anzahl der Bilder im Video
n_frames = int(T * fps)

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

# Skaliere das Signal so, dass es in den Wertebereich von
# ganzen 16-bit Zahlen passt (-32768 ... +32767) und wandle
# es anschließend in 16-bit-Integers um.
if np.max(np.abs(y)) > 0:
    y = np.int16(y / np.max(np.abs(y)) * 32767)

# Gib das Signal als Audiodatei im wav-Format aus.
scipy.io.wavfile.write('audio.wav', rate, y)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12.8, 7.2), dpi=150)
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.grid()

# Erzeuge zwei Kreise für Sender und Empfänger.
quelle, = ax.plot([0], [0], 'o', color='black')
beobach, = ax.plot([0], [0], 'o', color='blue')


def update(n):
    # Berechne, welcher audio_index zum i-ten Bild gehört.
    audio_index = int(n / fps * rate)

    # Aktualisiere die Positionen von Sender und Empfänger.
    quelle.set_data(r_B[audio_index])
    beobach.set_data(r_Q[audio_index])

    return quelle, beobach


# Erzeuge die Animation.
ani = mpl.animation.FuncAnimation(fig, update, frames=n_frames,
                                  interval=30, blit=True)

# Wir speichern jetzt die Animation ab. Wir haben bereits
# dafür gesorgt, dass die Dauer der Animation und die Dauer
# der Audiodatei gleich groß sind. Das eigentiche Erzeugen
# der Videodatei übernimmt das Programm ffmpeg. Diesem
# Programm müssen wir mit der Option "-i audio.wav" sagen,
# dass es einen zusätzlichen Eingabekanal gibt. Leider werden
# durch die zusätzliche Eingabeoption die Ausgabeoptionen für
# das Videoformat überschrieben. Wir müssen daher mit
# "-codec:v h264 -pix_fmt yuv420p" den Video-Codec und das
# Farbformat festlegen. Als nächstes geben wir mit "-codec:a
# aac" an, dass die Audiodaten im AAC-Codec gespeichert
# werden sollen. Die letzte Option "-crf 20" gibt die Qualität
# der Videodatei an. Je kleiner diese Zahl ist, desto besser
# ist die Qualität.
ani.save('output.mp4', fps=fps,
         extra_args=['-i', 'audio.wav',
                     '-codec:v', 'h264', '-pix_fmt', 'yuv420p',
                     '-codec:a', 'aac',
                     '-crf', '20'])

# Als letztes löschen wir die Wave-Datei.
os.unlink('audio.wav')

# Alternativ kann man auch mit
#    ani.save('video.mp4', fps=fps)
# nur eine Videodatei ohne Ton erzeugen. Anschließend kann man
# dann auf einer Kommandozeile die Audiodatei und die
# Videodatei zusammenführen. Dazu kann man auch wieder
# ffmpeg benutzen. Ein möglicher Kommandozeilenbefehl, der
# die Datei audio.wav und die Datei video.mp4 in eine
# Videodatei mit Ton überführt lautet:
#
#    ffmpeg -i audio.wav -i video.mp4 -codec:v copy -codec:a aac output.mp4
