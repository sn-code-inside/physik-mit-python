"""Dispersion eines gaußförmigen Signals.

Vergleich der Dispersion eines gaußförmigen Wellenpaketes
 a) Betrachtung im Teilchenmodell der Masse-Feder-Kette
 b) Betrachtung im Dispersionmodell. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

# Dimension des Raumes.
dim = 2

# Anzahl der Massen.
N = 100

# Federkonstante [N/m].
D = 100

# Masse [kg].
m = 0.05

# Länge der ungespannten Federn [m].
L0 = 0.05

# Abstand der Massen im ungespannten Zustand [m].
L0 = 0.15

# Abstand der Massen [m].
L = 0.15

# Amplitude [s].
A = 0.01

# Breite des Anregungspulses [s].
delta_t = 0.05


def teilchenmodell(t):
    """Gibt x und u zum Zeitpunkt t zurück. """

    # Ruhelage der N Massen im Abstand L auf der x-Achse.
    r0 = np.zeros((N, dim))
    r0[:, 0] = np.linspace(L, N * L, N)

    def anreg(t):
        """Ortsvektor der anregenden Masse zum Zeitpunkt t. """
        t_max = 3 * delta_t
        pos = np.empty(dim)
        pos[0] = A * np.exp(-((t - t_max) / delta_t) ** 2)
        return pos

    def federkraft(r1, r2):
        """Kraft auf die Masse am Ort r1. """
        L = np.linalg.norm(r2 - r1)
        F = D * (L - L0) * (r2 - r1) / L
        return F

    def dgl(t, u):
        r, v = np.split(u, 2)
        r = r.reshape(N, dim)
        a = np.zeros((N, dim))

        # Addiere die Beschleunigung durch die jeweils linke Feder.
        for i in range(1, N):
            a[i] += federkraft(r[i], r[i - 1]) / m

        # Addiere die Beschleunigung durch die jeweils rechte Feder.
        for i in range(N - 1):
            a[i] += federkraft(r[i], r[i + 1]) / m

        # Addiere die Beschleunigung durch die Anregende Masse.
        a[0] += federkraft(r[0], anreg(t)) / m

        # Die letzte Masse soll festgehalten werden.
        a[N - 1] = 0

        return np.concatenate([v, a.reshape(-1)])

    # Lege den Zustandsvektor zum Zeitpunkt t=0 fest. Alle N-1
    # Teilchen ruhen in der Ruhelage.
    v0 = np.zeros(N * dim)
    u0 = np.concatenate((r0.reshape(-1), v0))

    # Löse die Bewegungsgleichung bis zum Zeitpunkt t.
    result = scipy.integrate.solve_ivp(dgl, [0, t], u0)
    r, v = np.split(result.y, 2)

    # Wandle r in ein 3-dimensionals Array um:
    #    1. Index - Teilchennummer
    #    2. Index - Koordinatenrichtung
    #    3. Index - Zeitpunkt
    r = r.reshape(N, dim, -1)

    # Wir interessieren uns nur für die x-Koodinaten und die
    # relative Auslenkung aus der Ruhelage.
    x = r0[:, 0]
    u = r[:, 0, -1] - x

    return x, u


def dispersionsmodell(t):
    """Gibt x und u zum Zeitpunkt t zurück. """

    # Berechneter Bereich x = x_min ... x_max. Davon wird nur der
    # Bereich x = 0 ... x_max/2 in der Grafik dargestellt.
    x_max = 30.0

    # Zeitschrittweite [s] und Ortsauflösung [m].
    dx = 0.001

    # Mittlere Wellenzahl des Wellenpakets [1/m].
    k0 = 0.0

    # Breite des Wellenpakets [m] wird so gewählt, dass sie in
    # etwa dem Anregungsimpuls im Programm
    # kette_longitudinal1.py. Dazu müssen wir die zeitdauer
    # delta_t mit der Phasengeschwindigkeit multiplizieren.
    # entspricht.
    B = np.sqrt(D * L**2 / m) * delta_t

    # Im Teilchenmodell startet die Anregung bei x=0 zum
    # Zeitpunkt t=0. Wir verschieben den Puls hier also soweit
    # nach links, dass er zum Zeitpunkt t=0 gerade im
    # dargestellten Bereich auftaucht.Mittelpunkt des
    # Wellenpakets zum Zeitpunkt t=0 [m].
    x0 = -3 * B

    # Erzeuge je ein Array von x-Postionen.
    x = np.arange(-x_max, x_max, dx)

    # Lege die Wellenfunktion zum Zeitpunkt t=0 fest.
    u0 = A * np.exp(-((x - x0) / B) ** 2) * np.exp(1j * k0 * x)

    # Führe die Fourier-Transformation durch.
    u_ft = np.fft.fft(u0)

    # Berechne die zugehörigen Wellenzahlen.
    k = 2 * np.pi * np.fft.fftfreq(x.size, d=dx)

    # Implementiere die Dispersionsrelation. Wir müssen auch hier
    # wieder dafür sorgen, dass negative Wellenzahlen eine
    # negative Kreisfrequenz bekommen.
    omega = 2 * np.sqrt(D / m) * np.abs(
        np.sin(k * L / 2)) * np.sign(k)

    u = np.fft.ifft(u_ft * np.exp(-1j * omega * t))
    return x, np.real(u)


# Wir wollen bei Modell nach t = 1.5 s auswerten.
t = 1.5
x1, u1 = teilchenmodell(t)
x2, u2 = dispersionsmodell(t)

# Erzeuge eine Figure und eine Axes.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('x [m]')
ax.set_ylabel('Auslenkung [m]')
ax.set_xlim(0, 15)
ax.grid()

# Plotte beide Modellergebnisse.
ax.plot(x2, u2, '-', label='Dispersionsmodell')
ax.plot(x1, u1, 'o', label='Teilchenmodell')

# Zeige die Grafik an.
ax.legend()
plt.show()