"""Kurvenanpassung: Resonanzkurve. """

import math
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Anregungsfrequenz [Hz].
f = np.array([0.2, 0.5, 0.57, 0.63, 0.67,
              0.71, 0.80, 1.00, 1.33])

# Amplitude [cm].
A = np.array([0.84, 1.42, 1.80, 2.10, 2.22,
              2.06, 1.45, 0.64, 0.30])

# Fehler der Amplitude [cm].
dA = np.array([0.04, 0.07, 0.09, 0.11, 0.11,
               0.10, 0.08, 0.03, 0.02])


# Definition der Funktion, die an die Messdaten angepasst
# werden soll.
def func(f, A0, f0, delta):
    return A0 * f0**2 / np.sqrt(
        (f**2 - f0**2)**2 + (delta * f / math.pi)**2)


# Führe die Kurvenanpassung durch.
popt, pcov = scipy.optimize.curve_fit(func, f, A,
                                      [0.8, 0.7, 0.3], sigma=dA)
A0, f0, delta = popt
d_A0, d_f0, d_delta = np.sqrt(np.diag(pcov))

# Gib das Ergebnis der Kurvenanpassung aus.
print(f'Ergebnis der Kurvenanpassung:')
print(f'        A0 = ({A0:.2f} +- {d_A0:.2f}) cm.')
print(f'        f0 = ({f0:.3f} +- {d_f0:.3f}) Hz.')
print(f'     delta = ({delta:.2f} +- {d_delta:.2f}) 1/s')


# Erzeuge die Figure und ein Axes-Objekt mit logarithmischer
# Skalierung.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Anregungsfrequenz f [Hz]')
ax.set_ylabel('Amplitude A [cm]')
ax.grid()

# Plotte die angepasste Funktion mit einer hohen Auflösung.
f_fit = np.linspace(np.min(f), np.max(f), 500)
A_fit = func(f_fit, A0, f0, delta)
ax.plot(f_fit, A_fit, '-')

# Stelle die Messwerte dar.
ax.errorbar(f, A, yerr=dA, fmt='.', capsize=2)

# Zeige die Grafik an.
plt.show()
