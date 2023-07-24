"""Simulation des Stratosphärensprungs. Für die Luftdichte
werden tabellierte Messdaten geeignet interpoliert. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import scipy.interpolate

# Masse des Körpers [kg].
m = 90.0

# Erdbeschleunigung [m/s²].
g = 9.81

# Produkt aus c_w-Wert und Stirnfläche [m²].
cwA = 0.47

# Anfangshöhe [m].
y0 = 39.045e3

# Anfangsgeschwindigkeit [m/s].
v0 = 0.0

# Messwerte: Luftdichte [kg/m³] in Abhängigkeit von der Höhe [m].
h_mess = 1e3 * np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                         11.02, 15, 20.06, 25, 32.16, 40])
rho_mess = np.array([1.225, 1.112, 1.007, 0.909, 0.819, 0.736,
                     0.660, 0.590, 0.526, 0.467, 0.414, 0.364,
                     0.195, 0.0880, 0.0401, 0.0132, 0.004])

# Erzeuge eine Interpolationsfunktion für die Luftdichte.
fill = (rho_mess[0], rho_mess[-1])
rho = scipy.interpolate.interp1d(h_mess, rho_mess, kind='cubic',
                                 bounds_error=False,
                                 fill_value=fill)


def F(y, v):
    """Kraft als Funktion von Höhe y und Geschwindigkeit v. """
    Fg = -m * g
    Fr = -0.5 * rho(y) * cwA * v * np.abs(v)
    return Fg + Fr


def dgl(t, u):
    y, v = u
    return np.array([v, F(y, v) / m])


def aufprall(t, u):
    """Ereignisfunktion: liefert einen Vorzeichenwechsel beim
    Auftreffen auf dem Erdboden (y=0). """
    y, v = u
    return y


# Bende die Simulation beim Auftreten des Ereignisses.
aufprall.terminal = True

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.array([y0, v0])

# Löse die Bewegungsgleichung bis zum Auftreffen auf der Erde.
result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                   events=aufprall,
                                   dense_output=True)
t_s = result.t
y_s, v_s = result.y

# Berechne die Interpolation auf einem feinen Raster.
t = np.linspace(0, np.max(t_s), 1000)
y, v = result.sol(t)

# Erzeuge eine Figure.
fig = plt.figure(figsize=(9, 4))
fig.set_tight_layout(True)

# Plotte das Geschwindigkeits-Zeit-Diagramm.
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlabel('t [s]')
ax1.set_ylabel('v [m/s]')
ax1.grid()
ax1.plot(t_s, v_s, '.b')
ax1.plot(t, v, '-b')

# Plotte das Orts-Zeit-Diagramm.
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('y [m]')
ax2.grid()
ax2.plot(t_s, y_s, '.b')
ax2.plot(t, y, '-b')

# Zeige die Grafik an.
plt.show()
