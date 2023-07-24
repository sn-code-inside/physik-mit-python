"""Simulation eines fliegenden Balls mit Luftreibung.

In diesem Programm wird für verschiedene Abwurfgeschwindigkeiten
jeweils der optimale Abwurfwinkel bestimmt. Die sonstigen
Parameter entsprechen der Simulation des Tischtennisballs. """

import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.optimize

# Masse des Körpers [kg].
m = 2.7e-3

# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.45 * math.pi * 20e-3 ** 2

# Bereich von Abwurfgeschwindigkeit [m/s].
v0_min = 0.1
v0_max = 50

# Erdbeschleunigung [m/s²].
g = 9.81

# Luftdichte [kg/m³].
rho = 1.225

# Anfangshöhe [m].
h0 = 1.1


def bahnkurve(alpha, v0):
    """Liefert die Bahnkurve r(t) für einen schiefen Wurf mit
    Luftreibung. Zurückgegeben wird ein Array mit 1000
    Zeitpunkten und ein 2 x N - Array der Ortsvektoren. """

    # Anfangsort [m].
    r0 = np.array([0, h0])

    # Berechne den Vektor der Anfangsgeschwindigkeit [m/s].
    v0 = np.array([v0 * math.cos(math.radians(alpha)),
                   v0 * math.sin(math.radians(alpha))])

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
    u0 = np.concatenate((r0, v0))

    # Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde
    result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                       events=aufprall,
                                       dense_output=True)
    # Berechne die Interpolation auf einem feinen Raster.
    t = np.linspace(0, np.max(result.t), 1000)
    r, v = np.split(result.sol(t), 2)

    return t, r


def func(alpha, v0):
    """Berechne die Wurfweite und gib den Wert mit einem
    negativen Vorzeichen zurück. """
    t, r = bahnkurve(alpha, v0)
    # Die Wurfweite ist die x-Koordinate des letzten berechneten
    # Datenpunkts.
    return -r[0, -1]


# Erzeuge ein Array mit Anfangsgeschwindigkeiten.
v0 = np.linspace(v0_min, v0_max, 100)

# Erzeuge ein leeres Array, das für jede Anfangsgeschwindigkeit
# den optimalen Abwurfwinkel aufnimmt.
alpha = np.empty(v0.size)

# Führe für jeden Wert der Anfangsgeschwindigkeit die
# Optimierung durch. Die Geschwindigkeit muss als zusätzliches
# Argument an die Funktion func übergeben werden. Dies wird mit
# der Option arg=v bewirkt. Da der Winkel bei der Funktion func
# im Gradmaß angegeben wird, ist es sinnvoll, den Suchbereich
# mit bounds=(0, 90) auf den Bereich von 0 bis 90 Grad
# einzuschränken.
for i, v in enumerate(v0):
    result = scipy.optimize.minimize_scalar(func,
                                            bounds=(0, 90),
                                            args=v,
                                            method='bounded')
    alpha[i] = result.x

# Erzeuge eine Figure und ein Axes-Objekt.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Abwurfgeschwindigkeit [m/s]')
ax.set_ylabel('Optimaler Abwurfwinkel [°]')
ax.grid()

# Plott das Eregbnis.
ax.plot(v0, alpha)

# Zeige die Grafik an.
plt.show()
