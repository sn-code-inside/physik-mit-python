"""Gedämpfte Schwingung mit einer progressiven Feder:
Darstellung des Frequenzspektrums. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Zeitdauer der Simulation [s].
T = 1.0

# Abtastrate für die Tonwiedergabe [1/s].
rate = 44100

# Anfängliche Federhärte [N/m].
D0 = 400.0

# Konstante für den nichtlinearen Term [m].
alpha = 5e-3

# Masse [kg].
m = 1e-3

# Reibungskoeffizient [kg/s].
b = 0.01

# Anfangsauslenkung [m].
x0 = 20e-3

# Anfangsgeschwindigkeit [m/s].
v0 = 0


def dgl(t, u):
    x, v = np.split(u, 2)

    # Berücksichtige die nichtlineare Federkraft.
    F_f = -D0 * alpha * np.expm1(np.abs(x)/alpha) * np.sign(x)

    # Füge die geschwindigkeitsabhängige Reibungskraft hinzu.
    F = F_f - b * v

    return np.concatenate([v, F / m])


# Anfanszustand.
u0 = np.array([x0, v0])

# Löse die Differentialgleichung an den durch die Abtastrate
# vorgegebenen Zeitpunkten.
t = np.arange(0, T, 1 / rate)
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-6,
                                   t_eval=t)
x, v = result.y

# Führe die Fourier-Transformation durch.
x_ft = np.fft.fft(x) / x.size

# Bestimme die zugehörigen Frequenzen.
freq = np.fft.fftfreq(x.size, d=1/rate)

# Sortiere die Frequenzen in aufsteigender Reihenfolge.
freq = np.fft.fftshift(freq)
x_ft = np.fft.fftshift(x_ft)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(12, 4))
fig.set_tight_layout(True)

# Erezuge eine Axes und plotte den Zeitverlauf .
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Auslenkung')
ax1.grid()
ax1.plot(t, x)

# Erzeuge eine Axes und plotte den Betrag der
# Fourier-Transformierten.
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('Frequenz [Hz]')
ax2.set_ylabel('Amplitude')
ax2.set_xlim(-1000, 1000)
ax2.grid()
ax2.plot(freq, np.abs(x_ft))

# Zeige den Plot an.
plt.show()
