"""Kurvenanpassung mit Berücksichtung der Messfehler. """

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

# Dicke der Filter [mm].
d = np.array([0.000, 0.029, 0.039, 0.064, 0.136, 0.198,
              0.247, 0.319, 0.419, 0.511, 0.611, 0.719,
              0.800, 0.900, 1.000, 1.100, 1.189])

# Gemessene Intensität [Impulse / min].
n = np.array([2193, 1691, 1544, 1244, 706, 466,
              318, 202, 108, 80, 52, 47,
              45, 46, 47, 42, 43], dtype=float)

# Fehler der gemessenen Intensität [Impulse / min].
dn = np.array([47, 41, 39, 35, 26, 22,
               18, 14, 10, 9, 7, 7,
               7, 7, 7, 7, 7], dtype=float)


# Definition der Funktion, die gefittet werden soll.
def func(x, n_u, n_0, alpha):
    return n_u + n_0 * np.exp(-alpha * x)


# Führe die Kurvenanpassung durch.
popt, pcov = scipy.optimize.curve_fit(func, d, n,
                                      [40, 2200, 10], sigma=dn)
n_u, n_0, alpha = popt
d_n_u, d_n_0, d_alpha = np.sqrt(np.diag(pcov))

# Gib das Ergebnis der Kurvenanpassung aus.
print(f'Ergebnis der Kurvenanpassung:')
print(f'     n_u = ({n_u:4.0f} +- {d_n_u:2.0f}) 1/min.')
print(f'     n_0 = ({n_0:4.0f} +- {d_n_0:2.0f}) 1/min.')
print(f'   alpha = ({alpha:.2f} +- {d_alpha:.2f}) 1/mm.')

# Erzeuge die Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Filterdicke d [mm]')
ax.set_ylabel('Intensität n [1/min]')
ax.set_yscale('log')
ax.grid()

# Plotte die angepasste Funktion.
d_fit = np.linspace(np.min(d), np.max(d), 500)
n_fit = func(d_fit, n_u, n_0, alpha)
ax.plot(d_fit, n_fit, '-', zorder=2)

# Stelle die Messwerte dar.
ax.errorbar(d, n, yerr=dn, fmt='.', capsize=2, zorder=3)

plt.show()
