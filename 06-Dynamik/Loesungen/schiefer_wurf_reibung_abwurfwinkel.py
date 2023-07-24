"""Simulation eines fliegenden Balls mit Luftreibung.

In diesem Programm wird die Bahnkurve für verschiedene
Abwurfwinkel bei sonst festen Parametern dargestellt. Die
sonstigen Parameter entsprechen der Simulation des
Tischtennisballs. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Masse des Körpers [kg].
m = 2.7e-3

# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 20e-3 ** 2

# Betrag der Abwurfgeschwindigkeit [m/s].
v0 = 20

# Erdbeschleunigung [m/s²].
g = 9.81

# Luftdichte [kg/m³].
rho = 1.225

# Anfangshöhe [m].
h0 = 1.1


def bahnkurve(alpha):
    """Liefert die Bahnkurve r(t) für einen schiefen Wurf mit
    Luftreibung. Zurückgegeben wird ein Array mit 1000
    Zeitpunkten und ein 2xN-Array der Ortsvektoren. """

    # Anfangsort [m].
    r_0 = np.array([0, h0])

    # Berechne den Vektor der Anfangsgeschwindigkeit [m/s].
    v_0 = np.array([v0 * math.cos(alpha), v0 * math.sin(alpha)])

    def F(r, v):
        """Kraft als Funktion von Ort r und Geschwindigkeit v. """
        Fr = -0.5 * rho * cwA * np.linalg.norm(v) * v
        Fg = m * g * np.array([0, -1])
        return Fg + Fr

    def dgl(t, u):
        r, v = np.split(u, 2)
        return np.concatenate([v, F(r, v) / m])

    def aufprall(t, u):
        """Ereignisfunktion: Liefert einen Vorzeichenwechsel beim
        Auftreffen auf dem Erdboden (y=0). """
        r, v = np.split(u, 2)
        return r[1]

    # Beende die Simulation beim Auftreten des Ereignisses.
    aufprall.terminal = True

    # Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
    u0 = np.concatenate((r_0, v_0))

    # Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde
    result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                       events=aufprall,
                                       dense_output=True)

    # Berechne die Interpolation auf einem feinen Raster.
    t = np.linspace(0, np.max(result.t), 1000)
    r, v = np.split(result.sol(t), 2)

    return t, r


# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_aspect('equal')
ax.grid()

# Plotte die Bahnkurve für verschiedene Winkel.
for winkel in range(0, 70, 10):
    t, r = bahnkurve(math.radians(winkel))
    ax.plot(r[0], r[1], label=f'$\\alpha$ = {winkel}°')
ax.legend()

# Zeige die Grafik an.
plt.show()
