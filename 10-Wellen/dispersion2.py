"""Dispersion eines gaußförmigen Signals. """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation

# Simulationsdauer [s].
T = 250.0

# Berechneter Bereich x = 0 ... x_max. Davon wird nur der Bereich
# x = 0 ... x_max/2 in der Grafik dargestellt.
x_max = 2000.0

# Zeitschrittweite [s] und Ortsauflösung [m].
dt = 0.5
dx = 0.1

# Mittlere Wellenzahl des Wellenpakets [1/m].
k0 = 3.0

# Breite des Wellenpakets [m].
B = 12.5

# Mittelpunkt des Wellenpakets zum Zeitpunkt t=0 [m].
x0 = 14 * B

# Phasengeschwindigkeit bei k = k0 [m/s].
c0 = 10.0

# Differenz Gruppengeschwindigkeit - Phasengeschwindigkeit [m/s].
alpha = -5.0

# Erzeuge je ein Array von x-Postionen und Zeitpunkten.
x = np.arange(0, x_max, dx)
t = np.arange(0, T, dt)

# Lege die Wellenfunktion zum Zeitpunkt t=0 fest.
u0 = np.exp(-((x - x0) / B) ** 2) * np.exp(1j * k0 * x)

# Führe die Fourier-Transformation durch.
u_ft = np.fft.fft(u0)

# Berechne die zugehörigen Wellenzahlen.
k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)

# Berechne für jeden Wert von k die Phasengeschwindigkeit.
c = c0 + alpha * (np.abs(k) - k0) / k0

# Lege für jeden Wert von k die Kreisfrequenz fest.
omega = c * k

# Erzeuge eine Figure und eine Axes. Wir stellen dabei nur
# die erste Hälte der berechneten x-Werte dar.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('Auslenkung [a.u.]')
ax.set_xlim(0, x_max / 2)
ax.grid()

# Erzeuge einen Linienplot für die Welle.
welle0, = ax.plot(x, np.real(u0), color='lightblue')
welle, = ax.plot(x, np.real(u0), color='blue')

# Erzeuge ein Textfeld für die Ausgabe des Zeitpunktes.
text = ax.text(0.1, 0.9, '', transform=ax.transAxes)


def update(n):
    # Berechne die Wellenfunktion zum Zeitpunkt t[i].
    u = np.fft.ifft(u_ft * np.exp(-1j * omega * t[n]))
    welle.set_ydata(np.real(u))

    # Aktualisiere das Textfeld.
    text.set_text(f't = {t[n]:3.1f} s')

    return welle, text


# Erstelle die Animation und zeige die Grafik an.
ani = mpl.animation.FuncAnimation(fig, update, frames=t.size,
                                  interval=30, blit=True)
plt.show()
