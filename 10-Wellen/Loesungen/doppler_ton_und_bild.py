"""Hörbarmachen des Doppler-Effekts. Synchrone Audio- und
Videoausgabe mit Darstellung der Wellenausbreitung. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
import sounddevice

# Simulationsdauer [s].
T = 30.0

# Abtastrate [1/s].
rate = 44100

# Frequenz, mit der die Quelle Schallwellen aussendet [Hz].
f_Q = 300.0

# Frequenz, mit der die dargestellten Wellenzüge ausgesendet
# werden [Hz].
f_Q_graph = 1

# Ausbreitungsgeschwindigkeit der Welle [m/s].
c = 20.0

# Dargestellter Koordinatenbereich [m].
xlim = (-160, 160)
ylim = (-30, 30)

# Lege die Startposition der Quelle und des Beobachter fest [m].
r0_Q = np.array([-150.0, 5.0])
r0_B = np.array([0.0, -5.0])

# Lege die Geschwindigkeit von Quelle und Beobachter fest [m/s].
v_Q = np.array([15.0, 0.0])
v_B = np.array([0.0, 0.0])

# Erzeuge die Zeitachse und bestimme die Anzahl n der Zeitpunkte.
t = np.arange(0, T, 1/rate)

# Lege ein Array für das empfangene Signal an.
y = np.zeros(t.size)


def signal(t):
    """Ausgesendetes Signal als Funktion der Zeit. """
    sig = np.sin(2 * np.pi * f_Q * t)
    sig[t < 0] = 0.0
    return sig


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

# Erzeuge eine Figure und eine Axes.
fig = plt.figure(figsize=(12, 6))
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.grid()
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')

# Erzeuge zwei Kreise für Quelle und Beobachter.
quelle = mpl.patches.Circle((0, 0), radius=1.5,
                            color='black', fill=True, zorder=4)
beobach = mpl.patches.Circle((0, 0), radius=1.5,
                             color='blue', fill=True, zorder=4)
ax.add_patch(quelle)
ax.add_patch(beobach)

# Lege eine Liste an, die die kreisförmigen Wellenzüge speichert.
kreise = []

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
    outdata[dat.size:, 0] = 0.0

    # Erhöhe den Index um die Anzahl der verwendeten Datenpunkte.
    audio_index += min(dat.size, frames)


def update(n):
    """Aktualisiere die Grafik. """

    # Am Ende sollen der Sender und Empfänger stehen bleiben.
    # Ansonsten sollen die Positionen von Sender und Empfänger
    # synchron zu dem gerade ausgegebenen Audiosignal sein.
    n = min(audio_index, t.size - 1)

    # Aktualisiere die Position von Quelle und Beobachter.
    quelle.center= r_Q[n]
    beobach.center = r_B[n]

    # Erzeuge zum Startzeitpunkt einen neuen Kreis, oder wenn
    # seit dem Aussenden des letzten Wellenzuges mehr als eine
    # Periodendauer vergangen ist.
    if not kreise or t[n] >= kreise[-1].startzeit + 1 / f_Q_graph:
        kreis = mpl.patches.Circle(r_Q[n], radius=0,
                                   color='red', linewidth=1.5,
                                   fill=False, zorder=3)
        kreis.startzeit = t[n]
        kreise.append(kreis)
        ax.add_patch(kreis)

    # Aktualisiere die Radien aller dargestellen Kreise.
    for kreis in kreise:
        kreis.radius = (t[n] - kreis.startzeit) * c

    # Färbe den Beobachter rot, wenn ein Wellenzug auftrifft.
    beobach.set_color('blue')
    for kreis in kreise:
        d = np.linalg.norm(kreis.center - beobach.center)
        if abs(d - kreis.radius) < beobach.radius:
            beobach.set_color('red')

    # Färbe den Quelle rot, wenn ein Wellezug augesendet wird.
    d = np.linalg.norm(kreise[-1].center - quelle.center)
    if abs(d - kreise[-1].radius) < quelle.radius:
        quelle.set_color('red')
    else:
        quelle.set_color('black')

    return [quelle, beobach] + kreise


# Erzeuge einen Ausgabestrom für die Audioausgabe.
stream = sounddevice.OutputStream(rate, channels=1,
                                  callback=audio_callback)

# Erzeuge die Animation.
ani = mpl.animation.FuncAnimation(fig, update,
                                  interval=30, blit=True)

# Starte die Audioausgabe und die Animation.
with stream:
    plt.show(block=True)
