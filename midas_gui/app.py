import sys


def run():
    # Set Windows app ID so the taskbar shows our icon, not Python's
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("midas.gui")
    except (AttributeError, OSError):
        pass  # Not on Windows

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    # Load settings and set theme BEFORE importing widgets,
    # so their `from midas_gui.theme import THEME` gets the right value.
    from midas_gui.settings import Settings
    settings = Settings()

    from midas_gui.theme import set_theme, apply_theme
    set_theme(settings.theme_name)
    apply_theme(app)

    from midas_gui.main_window import MainWindow
    window = MainWindow(settings)
    window.show()
    return app.exec()
