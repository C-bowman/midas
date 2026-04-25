import sys
from PySide6.QtWidgets import QApplication
from midas_gui.main_window import MainWindow
from midas_gui.theme import apply_theme


def run():
    app = QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec()
