"""Gedämpfte Schwingung mit einer progressiven Feder. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.io.wavfile
import sounddevice

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
    F_f = -D0 * alpha * np.expm1(np.abs(x)/alpha) * np.sign(x)
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

# Skaliere das Signal so, dass es in den Wertebereich von
# ganzen 16-bit Zahlen passt (-32768 ... +32767) und wandle
# es anschließend in 16-bit-Integers um.
dat = np.int16(x / np.max(np.abs(x)) * 32767)

# Gib das Signal als Audiodatei im wav-Format aus.
scipy.io.wavfile.write('output.wav', rate, dat)

# Gib das Signal als Sound aus.
sounddevice.play(dat, rate, blocking=True)

# Erzeuge eine Figure und eine Axes und plotte die Kennlinien.
fig = plt.figure()
fig.set_tight_layout(True)
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('t [s]')
ax.set_ylabel('x [mm]')
ax.plot(t, x / 1e-3)

# Erzeuge eine Ausschnittsvergößerung.
axins1 = ax.inset_axes([0.55, 0.70, 0.4, 0.25])
axins1.plot(t, x / 1e-3)
axins1.set_xlim(0.0, 0.02)
axins1.set_ylim(-20.0, 20.0)
axins1.set_xlabel('t [s]')
axins1.set_ylabel('x [mm]')

# Erzeuge eine zweite Ausschnittsvergößerung.
axins2 = ax.inset_axes([0.55, 0.17, 0.4, 0.25])
axins2.plot(t, x / 1e-3)
axins2.set_xlim(0.8, 0.82)
axins2.set_ylim(-0.8, 0.8)
axins2.set_xlabel('t [s]')
axins2.set_ylabel('x [mm]')

# Zeige den Plot an.
plt.show()
