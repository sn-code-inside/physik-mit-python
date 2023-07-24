"""Beugungsfeld eines Spaltes durch Überlagerung von
kreisförmigen Wellen.

Hier: Verifikation anhand der Gleichung für die Intensität im
Fernfeld. """

import numpy as np
import matplotlib.pyplot as plt

# Betrachteter Abstand [m].
R = 500

# Wellenlänge [m].
lam = 1.0

# Anzahl der Elementarwellen.
N = 100

# Spaltbreite [m].
B = 10

# Lege die Startpositionen jeder Elementarwelle fest.
position = np.zeros((N, 2))
position[:, 0] = np.linspace(-B/2, B/2, N)

# Lege die Phase jeder Elementarwelle fest [rad].
phase = np.zeros(N)

# Lege die Amplitude jeder Elementarwelle fest.
amplitude = B / N * np.ones(N)

# Berechne die die Wellenzahl.
k = 2 * np.pi / lam

# Wir wollen die Wellenfunktionen nur auf einem Halbkreis mit
# Radius R auswerten.
alpha = np.radians(np.linspace(-90, 90, 1000))
x = R * np.sin(alpha)
y = R * np.cos(alpha)

# Wir legen nun ein Array komplexer Zahlen der passenden Größe
# an, das mit Nullen gefüllt ist.
u = np.zeros(x.shape, dtype=complex)

# Addiere die komplexe Amplitude jeder Elementarwelle.
for A, (x0, y0), phi0 in zip(amplitude, position, phase):
    d = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    cos_theta = (y - y0) / d
    u += A * np.exp(1j * (k * d + phi0)) / np.sqrt(d) * cos_theta

# Wir normieren die Intensität so, dass Sie beim Winkel alpha=0
# gerade den Wert 1 hat.
I_sim = np.abs(u)**2 / np.max(np.abs(u)**2)

# Fernfeldnäherung gemäß der angegebenen Gleichung mit I0 = 1
I_theo = np.sinc(B * np.sin(alpha) / lam) ** 2

# Erzeuge eine Figure und eine Axes und plotte beide Kurven.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlim(-20, 20)
ax.set_xlabel('$\\alpha$ [°]')
ax.set_ylabel('$I / I_0$')
ax.grid()

ax.plot(np.degrees(alpha), I_sim, label='Simulation')
ax.plot(np.degrees(alpha), I_theo, label='Fernfeldnäherung')

ax.legend()
plt.show()
