"""Kurvenanpassung: Strömungswiderstand. """

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Geschwindigkeit [m/s].
v = np.array([5.8, 7.3, 8.9, 10.6, 11.2])

# Fehler der Geschwindigkeit [m/s].
dv = np.array([0.3, 0.3, 0.2, 0.2, 0.1])

# Kraft [N].
F = np.array([0.10, 0.15, 0.22, 0.33, 0.36])

# Fehler der Kraft [N].
dF = np.array([0.02, 0.02, 0.02, 0.02, 0.02])


# Definition der Funktion, die an die Messdaten angepasst
# werden soll.
def func(v, b, n):
    return b * v ** n


# Führe die Kurvenanpassung durch.
popt, pcov = scipy.optimize.curve_fit(func, v, F,
                                      [1.5, 2.0], sigma=dF)
b, n = popt
db, dn = np.sqrt(np.diag(pcov))

# Gib das Ergebnis der Kurvenanpassung aus.
print(f'Ergebnis der Kurvenanpassung:')
print(f'     b = ({b:6.4f} +- {db:6.4f}) N / (m/s)^n.')
print(f'     n = {n:4.2f} +- {dn:2.2f}')

# Erzeuge die Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Geschwindigkeit v [m/s]')
ax.set_ylabel('Widerstandskraft F [N]')
ax.grid()

# Plotte die angepasste Funktion mit einer hohen Auflösung.
v_fit = np.linspace(np.min(v), np.max(v), 500)
F_fit = func(v_fit, b, n)
ax.plot(v_fit, F_fit, '-')

# Stelle die Messwerte dar.
ax.errorbar(v, F, xerr=dv, yerr=dF, fmt='.', capsize=2)

# Zeige die Grafik an.
plt.show()
