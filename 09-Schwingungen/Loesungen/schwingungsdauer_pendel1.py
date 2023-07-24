"""Schwingungsdauer des mathematischen Pendels.

Die Schwingungsdauer des mathematischen Pendels wird für große
Auslenkungswinkel numerisch bestimmt. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Anzahl der Winkel.
n = 200

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


# Werte die Funktion T_dgl für n Winkel zwischen phi_min und
# phi_max aus.
phi0 = np.linspace(phi_min, phi_max, n)
T_num = np.empty(n)
for i in range(n):
    T_num[i] = T_dgl(phi0[i])

# Plotte das Ergebnis.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Maximalauslenkung [°]')
ax.set_ylabel('Schwingungsdauer [s]')
ax.grid()
ax.plot(np.degrees(phi0), T_num)

# Zeige die Grafik an.
plt.show()
