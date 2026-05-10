from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QPushButton, QFileDialog,
    QFormLayout,
)
from PySide6.QtCore import Qt, Signal
import numpy as np

from midas_gui.theme import THEME


class ArrayEditor(QWidget):
    """Widget for entering array data via file import or generators."""

    value_changed = Signal()

    def __init__(self, label: str = "Array", parent=None):
        super().__init__(parent)
        self._values: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Source selector
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel(label))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["placeholder", "linspace", "arange", "constant", "file"])
        self.source_combo.currentTextChanged.connect(self._on_source_changed)
        source_row.addWidget(self.source_combo)
        layout.addLayout(source_row)

        # Linspace controls
        self.linspace_widget = QWidget()
        ls_layout = QFormLayout(self.linspace_widget)
        ls_layout.setContentsMargins(0, 0, 0, 0)
        self.ls_start = QDoubleSpinBox()
        self.ls_start.setRange(-1e12, 1e12)
        self.ls_start.setDecimals(4)
        self.ls_start.setValue(0.0)
        self.ls_stop = QDoubleSpinBox()
        self.ls_stop.setRange(-1e12, 1e12)
        self.ls_stop.setDecimals(4)
        self.ls_stop.setValue(1.0)
        self.ls_num = QSpinBox()
        self.ls_num.setRange(2, 100000)
        self.ls_num.setValue(10)
        ls_layout.addRow("Start:", self.ls_start)
        ls_layout.addRow("Stop:", self.ls_stop)
        ls_layout.addRow("Points:", self.ls_num)
        layout.addWidget(self.linspace_widget)

        for w in [self.ls_start, self.ls_stop, self.ls_num]:
            w.valueChanged.connect(self._generate)

        # Arange controls
        self.arange_widget = QWidget()
        ar_layout = QFormLayout(self.arange_widget)
        ar_layout.setContentsMargins(0, 0, 0, 0)
        self.ar_start = QDoubleSpinBox()
        self.ar_start.setRange(-1e12, 1e12)
        self.ar_start.setDecimals(4)
        self.ar_start.setValue(0.0)
        self.ar_stop = QDoubleSpinBox()
        self.ar_stop.setRange(-1e12, 1e12)
        self.ar_stop.setDecimals(4)
        self.ar_stop.setValue(1.0)
        self.ar_step = QDoubleSpinBox()
        self.ar_step.setRange(1e-8, 1e12)
        self.ar_step.setDecimals(4)
        self.ar_step.setValue(0.1)
        ar_layout.addRow("Start:", self.ar_start)
        ar_layout.addRow("Stop:", self.ar_stop)
        ar_layout.addRow("Step:", self.ar_step)
        layout.addWidget(self.arange_widget)
        self.arange_widget.hide()

        for w in [self.ar_start, self.ar_stop, self.ar_step]:
            w.valueChanged.connect(self._generate)

        # Full controls
        self.full_widget = QWidget()
        full_layout = QFormLayout(self.full_widget)
        full_layout.setContentsMargins(0, 0, 0, 0)
        self.full_size = QSpinBox()
        self.full_size.setRange(1, 100000)
        self.full_size.setValue(10)
        self.full_value = QDoubleSpinBox()
        self.full_value.setRange(-1e12, 1e12)
        self.full_value.setDecimals(4)
        self.full_value.setValue(0.0)
        full_layout.addRow("Size:", self.full_size)
        full_layout.addRow("Value:", self.full_value)
        layout.addWidget(self.full_widget)
        self.full_widget.hide()

        for w in [self.full_size, self.full_value]:
            w.valueChanged.connect(self._generate)

        # File controls
        self.file_widget = QWidget()
        file_layout = QHBoxLayout(self.file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)
        self.file_path_edit.setPlaceholderText("No file selected")
        file_layout.addWidget(self.file_path_edit)
        self._file_path: str = ""
        self._npz_key: str | None = None
        browse_btn = QPushButton("Browse\u2026")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)
        layout.addWidget(self.file_widget)
        self.file_widget.hide()

        # Preview label
        self.preview_label = QLabel("")
        self.preview_label.setStyleSheet(f"color: {THEME.text_secondary}; font-size: 10px;")
        self.preview_label.setWordWrap(True)
        layout.addWidget(self.preview_label)

        # Default to placeholder — hide all sub-widgets
        self.linspace_widget.hide()
        self.preview_label.setText("(no data — placeholder only)")

    def _on_source_changed(self, source: str):
        self.linspace_widget.setVisible(source == "linspace")
        self.arange_widget.setVisible(source == "arange")
        self.full_widget.setVisible(source == "constant")
        self.file_widget.setVisible(source == "file")
        if source == "placeholder":
            self._values = None
            self.preview_label.setText("(no data — placeholder only)")
            self.value_changed.emit()
        elif source != "file":
            self._generate()

    def _generate(self):
        source = self.source_combo.currentText()
        if source == "linspace":
            self._values = np.linspace(
                self.ls_start.value(), self.ls_stop.value(), self.ls_num.value()
            )
        elif source == "arange":
            self._values = np.arange(
                self.ar_start.value(), self.ar_stop.value(), self.ar_step.value()
            )
        elif source == "constant":
            self._values = np.full(self.full_size.value(), self.full_value.value())
        self._update_preview()
        self.value_changed.emit()

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Array", "",
            "NumPy files (*.npy *.npz);;CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        self.file_path_edit.setText(path)
        self._file_path = path
        self._npz_key = None
        try:
            if path.endswith(".csv"):
                self._values = np.loadtxt(path, delimiter=",").flatten()
            elif path.endswith(".npz"):
                data = np.load(path)
                keys = list(data.keys())
                if not keys:
                    self.preview_label.setText("Error: .npz file contains no arrays")
                    return
                elif len(keys) == 1:
                    key = keys[0]
                else:
                    from PySide6.QtWidgets import QInputDialog
                    key, ok = QInputDialog.getItem(
                        self, "Select Array",
                        "The .npz file contains multiple arrays.\nSelect which one to load:",
                        keys, 0, False,
                    )
                    if not ok:
                        return
                self._npz_key = key
                self._values = data[key].flatten()
                self.file_path_edit.setText(f"{path}  [{key}]")
            else:
                self._values = np.load(path).flatten()
            self._update_preview()
            self.value_changed.emit()
        except Exception as exc:
            self.preview_label.setText(f"Error: {exc}")

    def _update_preview(self):
        if self._values is None or len(self._values) == 0:
            self.preview_label.setText("(empty)")
            return
        v = self._values
        if len(v) <= 6:
            text = ", ".join(f"{x:.4g}" for x in v)
        else:
            head = ", ".join(f"{x:.4g}" for x in v[:3])
            tail = ", ".join(f"{x:.4g}" for x in v[-3:])
            text = f"{head}, …, {tail}"
        self.preview_label.setText(f"shape={v.shape}  [{text}]")

    @property
    def values(self) -> np.ndarray | None:
        return self._values

    def set_source(self, source: str):
        idx = self.source_combo.findText(source)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)

    def set_config(self, config: dict):
        """Restore the editor state from a config dict (as returned by get_config)."""
        source = config.get("source", "placeholder")
        self.set_source(source)
        if source == "linspace":
            self.ls_start.setValue(config.get("start", 0.0))
            self.ls_stop.setValue(config.get("stop", 1.0))
            self.ls_num.setValue(config.get("num", 10))
        elif source == "arange":
            self.ar_start.setValue(config.get("start", 0.0))
            self.ar_stop.setValue(config.get("stop", 1.0))
            self.ar_step.setValue(config.get("step", 0.1))
        elif source == "constant":
            self.full_size.setValue(config.get("size", 10))
            self.full_value.setValue(config.get("value", 0.0))
        elif source == "file":
            path = config.get("path", "")
            self._file_path = path
            self._npz_key = config.get("npz_key", None)
            if self._npz_key:
                self.file_path_edit.setText(f"{path}  [{self._npz_key}]")
            else:
                self.file_path_edit.setText(path)
        self._generate()

    def get_config(self) -> dict:
        source = self.source_combo.currentText()
        if source == "linspace":
            return {
                "source": "linspace",
                "start": self.ls_start.value(),
                "stop": self.ls_stop.value(),
                "num": self.ls_num.value(),
            }
        elif source == "arange":
            return {
                "source": "arange",
                "start": self.ar_start.value(),
                "stop": self.ar_stop.value(),
                "step": self.ar_step.value(),
            }
        elif source == "constant":
            return {
                "source": "constant",
                "size": self.full_size.value(),
                "value": self.full_value.value(),
            }
        elif source == "file":
            config = {
                "source": "file",
                "path": self._file_path,
            }
            if self._npz_key is not None:
                config["npz_key"] = self._npz_key
            return config
        else:
            return {"source": "placeholder"}
