"""Simulation des Sonnensystems.

Das Programm simuliert das Sonnensystem für einen Zeitraum von
50 Jahren und speichert die Ergebnisse in der Datei
ephemeriden.npz ab. """

import numpy as np
import scipy.integrate

# Dimension des Raumes.
dim = 3

# Ein Tag [s] und ein Jahr [s].
tag = 24 * 60 * 60
jahr = 365.25 * tag

# Eine Astronomische Einheit [m].
AE = 1.495978707e11

# Simulationszeitdauer T, Schrittweite dt [s].
T = 50 * jahr
dt = 5 * tag

# Newtonsche Graviationskonstante [m³ / (kg * s²)].
G = 6.674e-11

# Namen der simulierten Himmelskörper.
name = ['Sonne', 'Merkur', 'Venus', 'Erde',
        'Mars', 'Jupiter', 'Saturn', 'Uranus',
        'Neptun', '9P/Tempel 1', '2010TK7']

# Massen der Himmelskörper [kg].
# Quelle: https://ssd.jpl.nasa.gov/horizons.cgi
# Die Massen von 9P/Tempel 1 und 2017TK7 sind geschätzt.
m = np.array([1.9885e30, 3.302e23, 48.685e23, 5.9722e24,
              6.4171e23, 1.89813e27, 5.6834e26, 8.68103e25,
              1.02413e26, 7e13, 2e10])

# Positionen [m] und Geschwindigkeiten [m/s] der Himmelskörper
# am 01.01.2012 um 00:00 Uhr UTC.
# Quelle: https://ssd.jpl.nasa.gov/horizons.cgi
r0 = AE * np.array([
     [-3.241859398499088e-3, -1.331449770492458e-3, -8.441430972210388e-7],
     [-3.824910108111409e-1, -1.955727022061594e-1,  1.892637411059862e-2],
     [ 7.211147749926723e-1,  3.334025180138600e-2, -4.133082682493956e-2],
     [-1.704612905998195e-1,  9.676758607337962e-1, -3.140642423792612e-5],
     [-1.192725241298415e+0,  1.148990485621534e+0,  5.330857335041436e-2],
     [ 3.752622496696632e+0,  3.256207159994215e+0, -9.757709767384923e-2],
     [-8.943506571472968e+0, -3.720744112648929e+0,  4.206153526052092e-1],
     [ 2.003510615298455e+1,  1.205184752774219e+0, -2.550883982941838e-1],
     [ 2.601428919232999e+1, -1.493950125368399e+1, -2.918668092864814e-1],
     [ 3.092919273623052e+0,  6.374849521314798e-1, -4.938170253879825e-1],
     [-3.500888634488231e-1,  7.382457660686845e-1,  9.937175322228885e-2]])
v0 = AE / tag * np.array([
     [ 4.270617154820447e-6, -4.648506431568692e-6, -8.469657867642489e-8],
     [ 7.029877499006405e-3, -2.381780604663742e-2, -2.590381459216828e-3],
     [-1.045793358516289e-3,  2.010665107676625e-2,  3.360587977875350e-4],
     [-1.722905169624698e-2, -3.001024883870811e-3,  2.627266603191336e-7],
     [-9.195012836122981e-3, -8.871670960023885e-3,  4.000329706845314e-5],
     [-5.035554237289496e-3,  6.060385207824578e-3,  8.751352649528277e-5],
     [ 1.842572816910875e-3, -5.163338547394546e-3,  1.648327631319252e-5],
     [-2.649722266077889e-4,  3.742642248006496e-3,  1.735555169285604e-5],
     [ 1.542818728068733e-3,  2.740646317666675e-3, -9.236392136662484e-5],
     [ 3.234481019056261e-3,  8.932013115163753e-3,  3.798319697848072e-5],
     [-1.651037000673457e-2, -1.028444949884146e-2,  6.705542557361902e-3]])

# Anzahl der Himmelskörper.
N = len(name)

# Berechne die Schwerpunktsposition und -geschwindigkeit und
# ziehe diese von den Anfangsbedingungen ab.
r0 -= m @ r0 / np.sum(m)
v0 -= m @ v0 / np.sum(m)


def dgl(t, u):
    r, v = np.split(u, 2)
    r = r.reshape(N, dim)
    a = np.zeros((N, dim))
    for i in range(N):
        for j in range(i):
            dr = r[j] - r[i]
            gr = G / np.linalg.norm(dr) ** 3 * dr
            a[i] += gr * m[j]
            a[j] -= gr * m[i]
    return np.concatenate([v, a.reshape(-1)])


# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0.reshape(-1), v0.reshape(-1)))

# Löse die Bewegungsgleichung bis zum Zeitpunkt T.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, rtol=1e-9,
                                   t_eval=np.arange(0, T, dt))
t = result.t
r, v = np.split(result.y, 2)

# Wandle r und v in ein 3-dimensionals Array um:
#    1. Index - Himmelskörper
#    2. Index - Koordinatenrichtung
#    3. Index - Zeitpunkt
r = r.reshape(N, dim, -1)
v = v.reshape(N, dim, -1)

# Speichere die Simulationsdaten ab.
np.savez('ephemeriden.npz',
         tag=tag, jahr=jahr, G=G, AE=AE, T=T, dt=dt,
         m=m, t=t, r=r, v=v, name=name)
