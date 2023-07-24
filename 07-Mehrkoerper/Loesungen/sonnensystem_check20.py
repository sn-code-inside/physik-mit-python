"""Vergleich der Positionen einiger Himmelskörper in der
Simulation mit den Daten aus der Horizonts-Datenbank nach einer
simulierten Zeitdauer von 20 Jahren. """

import datetime
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

# Lies die Simulationsdaten ein.
dat = np.load('ephemeriden.npz')
tag, jahr, AE, G = dat['tag'], dat['jahr'], dat['AE'], dat['G']
T, dt = dat['T'], dat['dt']
m, name = dat['m'], dat['name']
t, r, v = dat['t'], dat['r'],  dat['v']

# Farben für die Darstellung der Planetenbahnen.
farbe = ['gold', 'darkcyan', 'orange', 'blue', 'red', 'brown',
         'olive', 'green', 'slateblue', 'black', 'gray']

# Anzahl der Himmelskörper.
N = m.size

# Positionen [m] und Geschwindigkeiten [m/s] der Himmelskörper
# am 01.01.2032 um 00:00 Uhr UTC.
# Quelle: https://ssd.jpl.nasa.gov/horizons.cgi
# Sonne, Merkur, Venus, Erde, Mars, Jupiter, Saturn, Uranus,
# Neptun, Tempel1, 2010TK7
r0 = AE * np.array([
    [-2.873582453644109E-03, 1.147258219149997E-03, 1.009976029867639E-04],
    [-3.485156269131940E-01, -2.725861224999921E-01, 9.425454379825622E-03],
    [-7.160897748475414E-01, -9.340398627979012E-02, 3.994942706992712E-02],
    [-1.678546481444298E-01, 9.704687622963032E-01, 3.496195160532046E-05],
    [1.387729812783350E+00, 1.215121559400403E-01, -3.145927307889373E-02],
    [8.984532834372694E-01, -5.130146026796059E+00, 1.266296518467286E-03],
    [1.590630789906994E+00, 8.882729756913912E+00, -2.178061828452681E-01],
    [1.677906144539834E+00, 1.901094031211258E+01, 4.884543244780725E-02],
    [2.891338264091845E+01, 7.355991713006560E+00, -8.178229347004724E-01],
    [-3.017365665281814E-01, 4.509581182632648E+00, 3.816681600888835E-01],
    [-5.040752228290852E-01, 6.169753242151466E-01, 1.642740160622794E-01]])
v0 = AE / tag * np.array([
    [-5.097638224126704E-06, -1.968167344847541E-06, 1.038034257063109E-07],
    [1.170771864371389E-02, -2.080263167327923E-02, -2.774052910460282E-03],
    [2.527222668370369E-03, -2.014270248912113E-02, -4.230076634418595E-04],
    [-1.724161836433711E-02, -2.949059719801611E-03, 4.483196824745523E-07],
    [-6.764915715900500E-04, 1.513457871367736E-02, 3.337853455581056E-04],
    [7.343768002116332E-03, 1.657125882047669E-03, -1.711739316235691E-04],
    [-5.791513821185702E-03, 9.743730798176529E-04, 2.134391414877257E-04],
    [-3.946799516501498E-03, 1.618225968806492E-04, 5.162262085100514E-05],
    [-7.939412580164786E-04, 3.060797443310768E-03, -4.512886202747095E-05],
    [-5.890261558094339E-03, -2.381676402461833E-03, 8.216517517470928E-04],
    [-1.418747446662007E-02, -1.402087997368813E-02, 5.980823402832862E-03]])

# Berechne die Schwerpunktsposition und -geschwindigkeit und
# ziehe diese von den Anfangsbedingungen ab.
r0 -= m @ r0 / np.sum(m)
v0 -= m @ v0 / np.sum(m)

# Zwischen dem 01.01.2012 und dem 01.01.2023 sind 7305 Tage
# vergangen. Das kann man mit dem Python-Modul datetime
# ausrechnen. Dazu werden zwei Objekte vom Typ datetime.date
# erzeugt. Wenn man diese zwei Objekte voneinander abzieht
# erhält man ein Objekt vom Typ datetime.timedelta. Diese
# Objekt besitzt eine Methode total_seconds, mit der man die
# Zeitdifferenz in Sekunden berechnen kann. Dabei werden
# automatisch eventuelle Schalttage berücksichtigt,
# Schaltsekunden werden allerdings ignoriert.
datum1 = datetime.date(2012, 1, 1)
datum2 = datetime.date(2032, 1, 1)
delta_t = (datum2 - datum1).total_seconds()

# Suche den Index in den Simulationsdaten, der am nächsten am
# gesuchten Zeitpunkt liegt.
k = np.argmin(np.abs(t - delta_t))

# Erzeuge eine Figure und eine 3D-Axes.
fig = plt.figure(figsize=(9, 6))
fig.tight_layout()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.set_xlabel('x [AE]')
ax.set_ylabel('y [AE]')
ax.set_zlabel('z [AE]')
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([-3, 3])
ax.grid()

# Plotte für jeden Planeten die Bahnkurve und füge die
# Beschriftungen hinzu. Wir reduzieren die Linienstärke, damit
# man die aktuelle Position der Himmelskörper besser erkennen
# kann.
for i in range(N):
    ax.plot(r[i, 0, :] / AE, r[i, 1, :] / AE, r[i, 2, :] / AE,
            '-', color=farbe[i], linewidth=0.2)

# Plotte die simulierten Positionen der Himmelskörper mit
# offenen Kreisen.
for i in range(N):
    ax.plot([r[i, 0, k] / AE],
            [r[i, 1, k] / AE],
            [r[i, 2, k] / AE], 'o',
            fillstyle='none', color=farbe[i], label=name[i])
ax.legend()

# Plotte die Positionen der Himmelskörper aus der Horizons-
# Datenbank mit offenen Rauten (engl. diamond).
for i in range(N):
    ax.plot([r0[i, 0] / AE],
            [r0[i, 1] / AE],
            [r0[i, 2] / AE], 'D',
            fillstyle='none', color=farbe[i])

# Zeige den Plot an.
plt.show()

