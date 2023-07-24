"""Schwingungsdauer des mathematischen Pendels.

Die Schwingungsdauer des mathematischen Pendels wird für große
Auslenkungswinkel mit der Intragralformel bestimmt und mit dem
Ergebnis aus der numerischen Lösung der Differentialgleichung
verglichen. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Anzahl der Winkel
n = 1000

# Minimaler und maximaler Auslenkwinkel [rad].
phi_min = math.radians(1)
phi_max = math.radians(175)

# Die Erdbeschleunigung g [m/s²] und die Pendellänge L [m]
# werden so gewählt, dass sich für kleine Auslenkungen eine
# Schiwngungsdauer von 1 s ergibt.
g = 9.81
L = g / (2 * math.pi) ** 2


def dgl(t, u):
    phi, dphi = u
    return [dphi, - g / L * np.sin(phi)]


def nulldurchgang(t, u):
    """Abbruchkriterium für die Lösung der Dgl. """
    phi, dphi = u
    return phi


# Wir starten mit einer Auslenkung in positiver Richtung
# und ohne Anfangsgeschwindigktei. Beim ersten
# Nulldurchgang des Winkels in negativer Richtung ist
# gerade 1/4 Periode vorbei.
nulldurchgang.terminal = True
nulldurchgang.direction = -1


def T_dgl(phi0):
    """Über die Lösung der Differentialgleichung bestimmte
    Schwingungsdauer [s] für den Maximalwinkel phi0 [rad]. """
    # Integriere die Differentialgleichung numerisch bis zum
    # ersten Nulldurchgang.
    u0 = np.array([phi0, 0])
    result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                       events=nulldurchgang)

    # Es wurde über eine viertel Periode integriert.
    return 4 * result.t[-1]


def f(phi, phi_max):
    """Integrand zur Berechnung der Periodendauer. """
    a = np.sqrt(2 * L / g)
    return a / np.sqrt(np.cos(phi) - np.cos(phi_max))


# Werte die Funktion T_dgl für n Winkel zwischen phi_min und
# phi_max aus.
phi0 = np.linspace(phi_min, phi_max, n)
T_num = np.empty(n)
for i in range(n):
    T_num[i] = T_dgl(phi0[i])

# Werte das Integral an den entsprechenden Winkeln aus.
T_int = np.empty(n)
for i in range(n):
    T_int[i], err = scipy.integrate.quad(f, -phi0[i], phi0[i],
                                         args=(phi0[i],))

# Erzeuge eine Figure.
fig = plt.figure()
fig.set_tight_layout(True)

# Erzeuge eine Axes und plotte die berechneten Schwingungsdauern
# als Funktion der Auslenkung.
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Maximalauslenkung [°]')
ax.set_ylabel('Schwingungsdauer [s]')
ax.grid()
ax.plot(np.degrees(phi0), T_num, 'b--', zorder=5,
        label='num. Lösung der Dgl.')
ax.plot(np.degrees(phi0), T_int, 'r-', zorder=4,
        label='Integralformel')
ax.legend(loc='upper left')

# Erzeuge eine zweite Axes und plotte die relative Abweichung
# der beiden Ergebnisse.
ax2 = ax.twinx()
ax2.set_ylabel('Relative Abweichung [ppm]', color='green')
ax2.tick_params(axis='y', labelcolor='green')
ax2.plot(np.degrees(phi0), 1e6 * (T_int - T_num) / T_num,
         'g', zorder=3, label='rel. Abw.')
ax2.legend(loc='upper right')

# Zeige die Grafik an.
plt.show()