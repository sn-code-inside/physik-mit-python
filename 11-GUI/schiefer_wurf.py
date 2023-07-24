"""GUI-Applikation zum schiefen Wurf mit Luftreibung. """

# Importiere Matplotlib.
import matplotlib as mpl
import matplotlib.backends.backend_qt5agg
import matplotlib.figure

# Importiere die notwendigen Elemente für die GUI.
import PyQt5
import PyQt5.uic
import PyQt5.QtWidgets

# Importiere sonstige Module.
import math
import numpy as np
import scipy.integrate


# Definiere eine Klasse, die von QMainWindow abgeleitet wird.
class MainWindow(PyQt5.QtWidgets.QMainWindow):

    def __init__(self):
        # Initialisiere das QMainWindow.
        super().__init__()

        # Lade das Benutzerinterface und erzeuge die GUI-Objekte.
        PyQt5.uic.loadUi('schiefer_wurf.ui', self)

        # Definiere eine Statusvariable, die bei einer
        # fehlerhaften Eingabe auf False gesetzt wird.
        self.eingabe_okay = True

        # Erzeuge eine Figure und eine Qt-Zeichenfläche (Canvas).
        self.fig = mpl.figure.Figure()
        mpl.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig)
        self.fig.set_tight_layout(True)

        # Füge die Zeichenfläche (canvas) in die GUI ein.
        self.box_plot.layout().addWidget(self.fig.canvas)

        # Erzeuge eine Axes.
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('x [m]')
        self.ax.set_aspect('equal')
        self.ax.grid()

        # Erzeuge einen Linienplot für die Bahnkurve.
        self.plot, = self.ax.plot([], [])

        # Wenn sich der Wert des Sliders ändert, dann soll auch
        # das Feld mit dem numerischen Wert des Winkels
        # aktualisiert werden.
        self.slider_alpha.valueChanged.connect(self.winkelanzeige)

        # Wenn einer der Eingabewerte verändert wird, dann soll
        # automatisch eine neue Simulation gestartet werden.
        self.slider_alpha.valueChanged.connect(self.simulation)
        self.edit_h.editingFinished.connect(self.simulation)
        self.edit_v.editingFinished.connect(self.simulation)
        self.edit_m.editingFinished.connect(self.simulation)
        self.edit_cwArho.editingFinished.connect(self.simulation)
        self.edit_g.editingFinished.connect(self.simulation)
        self.edit_xmax.editingFinished.connect(self.simulation)
        self.edit_ymax.editingFinished.connect(self.simulation)

        # Starte erstmalig die Simulation.
        self.winkelanzeige()
        self.simulation()

    def winkelanzeige(self):
        """Aktualisiert das Feld für die Winkelangabe. """
        alpha = self.slider_alpha.value()
        self.label_alpha.setText(f'{alpha}°')

    def eingabe_float(self, field):
        """Lies eine Gleitkommazahl aus einem Textfeld aus. """
        try:
            value = float(field.text())
        except ValueError:
            self.eingabe_okay = False
            field.setStyleSheet("border: 2px solid red")
            self.statusbar.showMessage('Fehlerhafte Eingabe!')
        else:
            field.setStyleSheet("")
            return value

    def simulation(self):
        # Setze die Statusvariable zurück.
        self.eingabe_okay = True

        # Lies die Parameter aus den Eingabefeldern.
        hoehe = self.eingabe_float(self.edit_h)
        geschw = self.eingabe_float(self.edit_v)
        m = self.eingabe_float(self.edit_m)
        cwArho = self.eingabe_float(self.edit_cwArho)
        g = self.eingabe_float(self.edit_g)
        xmax = self.eingabe_float(self.edit_xmax)
        ymax = self.eingabe_float(self.edit_ymax)
        alpha = math.radians(self.slider_alpha.value())

        # Überprüfe, ob alle Eingaben gültig sind.
        if not self.eingabe_okay:
            return

        # Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
        r0 = np.array([0, hoehe])
        v0 = geschw * np.array([math.cos(alpha), math.sin(alpha)])
        u0 = np.concatenate((r0, v0))

        def dgl(t, u):
            """Rechte Seite des Differentialgleichungssystems. """
            r, v = np.split(u, 2)
            # Luftreibungskraft.
            Fr = -0.5 * cwArho * np.linalg.norm(v) * v
            # Schwerkraft.
            Fg = m * g * np.array([0, -1])
            # Beschleunigung.
            a = (Fr + Fg) / m
            return np.concatenate([v, a])

        def aufprall(t, u):
            """Ereignisfunktion: Aufprall auf dem Erdboden. """
            r, v = np.split(u, 2)
            return r[1]

        # Beende die Integration, wenn der Gegenstand auf dem
        # Boden aufkommt.
        aufprall.terminal = True
        aufprall.direction = -1

        # Löse die Bewegungsgleichung.
        result = scipy.integrate.solve_ivp(dgl, [0, np.inf], u0,
                                           events=aufprall,
                                           dense_output=True)

        # Berechne die Interpolation auf einem feinen Raster.
        t = np.linspace(0, np.max(result.t), 1000)
        r, v = np.split(result.sol(t), 2)

        # Aktualisiere den Plot.
        self.plot.set_data(r[0], r[1])
        self.ax.set_xlim(0, xmax)
        self.ax.set_ylim(0, ymax)

        # Zeichne die Grafikelemente neu.
        self.fig.canvas.draw()

        # Lösche den Statustext von der vorherigen Simulation.
        self.statusbar.clearMessage()


# Erzeuge eine QApplication und das Hauptfenster.
app = PyQt5.QtWidgets.QApplication([])
window = MainWindow()

# Zeige das Fenster und starte die QApplication.
window.show()
app.exec_()
