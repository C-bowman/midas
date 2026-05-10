from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QSpinBox,
    QDialogButtonBox, QGroupBox, QComboBox, QMessageBox,
)

from midas_gui.theme import THEMES

_DEFAULTS = {
    "font_size/palette": 12,
    "font_size/properties": 12,
    "font_size/code_preview": 12,
}


class Settings(QObject):
    """Persistent application settings backed by QSettings."""

    font_size_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._qs = QSettings("MIDAS", "midas_gui")

    def _get(self, key: str) -> int:
        return int(self._qs.value(key, _DEFAULTS[key]))

    def _set(self, key: str, value: int):
        self._qs.setValue(key, value)

    @property
    def palette_font_size(self) -> int:
        return self._get("font_size/palette")

    @palette_font_size.setter
    def palette_font_size(self, value: int):
        self._set("font_size/palette", value)

    @property
    def properties_font_size(self) -> int:
        return self._get("font_size/properties")

    @properties_font_size.setter
    def properties_font_size(self, value: int):
        self._set("font_size/properties", value)

    @property
    def code_preview_font_size(self) -> int:
        return self._get("font_size/code_preview")

    @code_preview_font_size.setter
    def code_preview_font_size(self, value: int):
        self._set("font_size/code_preview", value)

    @property
    def theme_name(self) -> str:
        return str(self._qs.value("theme/name", "Deep Ocean"))

    @theme_name.setter
    def theme_name(self, value: str):
        self._qs.setValue("theme/name", value)


class SettingsDialog(QDialog):
    """Modal dialog for editing application settings."""

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(350)
        self._settings = settings

        layout = QVBoxLayout(self)

        # Theme group
        theme_group = QGroupBox("Appearance")
        theme_form = QFormLayout(theme_group)

        self._theme_combo = QComboBox()
        for name in THEMES:
            self._theme_combo.addItem(name)
        self._theme_combo.setCurrentText(settings.theme_name)
        theme_form.addRow("Theme:", self._theme_combo)

        layout.addWidget(theme_group)

        # Font sizes group
        group = QGroupBox("Font Sizes")
        form = QFormLayout(group)

        self._palette_spin = self._make_spin(settings.palette_font_size)
        form.addRow("Node Palette:", self._palette_spin)

        self._props_spin = self._make_spin(settings.properties_font_size)
        form.addRow("Properties Panel:", self._props_spin)

        self._code_spin = self._make_spin(settings.code_preview_font_size)
        form.addRow("Code Preview:", self._code_spin)

        layout.addWidget(group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._apply_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_spin(value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(6, 24)
        spin.setSuffix(" pt")
        spin.setValue(value)
        return spin

    def _apply_and_accept(self):
        self._settings.palette_font_size = self._palette_spin.value()
        self._settings.properties_font_size = self._props_spin.value()
        self._settings.code_preview_font_size = self._code_spin.value()
        self._settings.font_size_changed.emit()

        new_theme = self._theme_combo.currentText()
        if new_theme != self._settings.theme_name:
            self._settings.theme_name = new_theme
            QMessageBox.information(
                self, "Theme Changed",
                "The new theme will be applied when you restart the application.",
            )

        self.accept()
