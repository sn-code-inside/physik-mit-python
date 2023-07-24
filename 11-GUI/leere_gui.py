"""GUI mit PyQT ohne irgendwelche Funktionalität. """
import PyQt5.QtWidgets

# Erzeuge eine QApplication und das Hauptfenster.
app = PyQt5.QtWidgets.QApplication([])
window = PyQt5.QtWidgets.QMainWindow()

# Zeige das Fenster und starte die QApplication.
window.show()
app.exec_()
