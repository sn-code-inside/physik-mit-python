"""Halblogarithmische Darstellung einer Resonanzkurve. """

import numpy as np
import matplotlib.pyplot as plt

# Eigenkreisfrequenz des Systems [1/s].
omega0 = 1.0

# Abklingkoeffizienten [1/s].
deltas = np.array([0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2])

# Anregungsfrequenzen [Hz].
omega = np.logspace(-1, 1, 500)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(10, 5))
fig.set_tight_layout(True)

# Erzeuge eine Axes für die Amplitude.
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xscale('log')
ax1.set_xlabel('$\\omega / \\omega_0$')
ax1.set_ylabel('Amplitude')
ax1.set_ylim(0, 6)
ax1.grid()

# Erzeuge eine zweite Axes für die Phase.
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xscale('log')
ax2.set_xlabel('$\\omega / \\omega_0$')
ax2.set_ylabel('Phasenverschiebung [rad]')
ax2.grid()

# Plotte den Amplituden- und Phasenverlauf als Funktion der
# Anregungskreisfrequenz.
for delta in deltas:
    x = omega0 ** 2 / (
            omega0 ** 2 - omega ** 2 + 2 * 1j * delta * omega)
    labeltext = f'$\\delta$={delta/omega0:.1f}'
    ax1.plot(omega / omega0, np.abs(x), label=labeltext)
    ax2.plot(omega / omega0, -np.angle(x), label=labeltext)

# Erzeuge nur für die erste Axes eine Legende.
ax1.legend()

# Zeige die Figure an.
plt.show()
