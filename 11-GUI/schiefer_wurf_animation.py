"""GUI-Applikation zum schiefen Wurf mit Luftreibung. Die
Bahnkurve wird mit einer Animation der Bewegung dargestellt. """

# Importiere Matplotlib.
import matplotlib as mpl
import matplotlib.backends.backend_qt5agg
import matplotlib.figure
import matplotlib.animation

# Importiere die notenwendigen Elemente für die GUI.
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
        PyQt5.uic.loadUi('schiefer_wurf_animation.ui', self)

        # Definiere eine Statusvariable, die bei eine
        # fehlerhaften Eingabe auf False gesetzt wird.
        self.eingabe_okay = True

        # Erzeuge eine Figure und eine Axes.
        self.fig = mpl.figure.Figure()
        self.fig.set_tight_layout(True)

        # Erzeuge einen Zeichenfläche (Canvas). Diese wird
        # automatisch mit der Figure verbunden.
        mpl.backends.backend_qt5agg.FigureCanvasQTAgg(self.fig)

        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_xlabel('x [m]')
        self.ax.set_ylabel('x [m]')
        self.ax.set_aspect('equal')
        self.ax.grid()

        # Füge die Zeichenfläche (canvas) in die GUI ein.
        self.box_plot.layout().addWidget(self.fig.canvas)

        # Erzeuge einen Linienplot für die Bahnkurve.
        self.plot, = self.ax.plot([0], [0], zorder=5)

        # Erzeuge einen Punktplot für die aktuelle Position.
        self.punkt, = self.ax.plot([], [], 'or', zorder=6)

        # Erzeuge ein Textfeld zur Ausgabe der aktuellen Zeit.
        self.text = self.ax.text(0.01, 0.99, '',
                                 verticalalignment='top',
                                 transform=self.ax.transAxes)

        # Initialisiere die Attribute für das Simulationsergebnis.
        self.t = np.zeros(0)
        self.r = np.zeros((2, 0))
        self.v = np.zeros((2, 0))

        # Update-Funktion für die Animation
        def update(n):
            if n < self.t.size:
                self.punkt.set_data(self.r[0, n], self.r[1, n])
                self.text.set_text(f't = {self.t[n]:.2f} s')
            else:
                # Halte die Animation an, wenn wir am Ende
                # der Simulation angekommen sind.
                self.punkt.set_data([], [])
                self.text.set_text('')
                self.anim.event_source.stop()
            return self.punkt, self.text, self.plot

        # Setze das in der Animation als nächstes angezeigte
        # Bild auf Null.
        self.currentframe = 0

        # Definiere einen Generator. Dieser gibt die aktuelle
        # Bildnummer zurück und erhöht diese so lange, bis das
        # Ende der Simulation erreicht ist.
        def frames():
            while True:
                yield self.currentframe
                if self.currentframe < self.t.size:
                    self.currentframe += 1

        # Starte die Iteration und übergib den Generator.
        self.anim = mpl.animation.FuncAnimation(self.fig, update,
                                                frames=frames(),
                                                interval=30,
                                                blit=True,
                                                repeat=False)

        # Starte erstmalig die Simulation.
        self.winkelanzeige()
        self.simulation()

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

        self.button_start.clicked.connect(self.start_animation)
        self.button_stop.clicked.connect(self.stop_animation)

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

    def start_animation(self):
        self.currentframe = 0
        self.anim.event_source.start()

    def stop_animation(self):
        self.currentframe = self.t.size

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
        v0 = geschw * np.array(
            [math.cos(alpha), math.sin(alpha)])
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
        result = scipy.integrate.solve_ivp(dgl, [0, np.inf],
                                           u0,
                                           events=aufprall,
                                           dense_output=True)

        # Berechne die Interpolation auf einem feinen Raster.
        self.t = np.linspace(0, np.max(result.t), 1000)
        self.r, self.v = np.split(result.sol(self.t), 2)

        # Aktualisiere den Plot.
        self.plot.set_data(self.r[0], self.r[1])
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
