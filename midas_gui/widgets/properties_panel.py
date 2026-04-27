from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QSpinBox,
    QDoubleSpinBox, QComboBox, QFormLayout, QScrollArea,
    QFrame, QPushButton, QFileDialog,
)
from PySide6.QtCore import Qt, Signal

from midas_gui.session import NodeModel, NODE_TYPES
from midas_gui.theme import THEME
from midas_gui.widgets.array_editor import ArrayEditor
from midas_gui.settings import Settings


class PropertiesPanel(QWidget):
    """Right sidebar: edits properties of the currently selected node."""

    node_updated = Signal(str)  # emits node_id
    ports_changed = Signal(str)  # emits node_id when dynamic ports need rebuild

    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self._node: NodeModel | None = None
        self._editors: dict[str, QWidget] = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        self._header = QLabel("Properties")
        self._header.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {THEME.text_primary}; padding: 4px;"
        )
        outer.addWidget(self._header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll)

        self._form_container = QWidget()
        container_layout = QVBoxLayout(self._form_container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        self._form_widget = QWidget()
        self._form_layout = QFormLayout(self._form_widget)
        self._form_layout.setContentsMargins(4, 4, 4, 4)
        self._form_layout.setSpacing(8)
        self._form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        container_layout.addWidget(self._form_widget)
        container_layout.addStretch(1)
        scroll.setWidget(self._form_container)

        self._placeholder = QLabel("Select a node to edit its properties.")
        self._placeholder.setStyleSheet(f"color: {THEME.text_secondary}; padding: 16px;")
        self._placeholder.setWordWrap(True)
        self._form_layout.addRow(self._placeholder)

        self._apply_font()
        settings.font_size_changed.connect(self._apply_font)

    def _apply_font(self):
        from PySide6.QtGui import QFont
        self._form_widget.setFont(QFont("Segoe UI", self._settings.properties_font_size))

    def set_node(self, node: NodeModel | None):
        self._node = node
        self._clear_form()

        if node is None:
            self._header.setText("Properties")
            self._placeholder = QLabel("Select a node to edit its properties.")
            self._placeholder.setStyleSheet(f"color: {THEME.text_secondary}; padding: 16px;")
            self._placeholder.setWordWrap(True)
            self._form_layout.addRow(self._placeholder)
            return

        spec = node.spec
        self._header.setText(f"{spec.display_name}")

        self._build_editors(node)

    def _clear_form(self):
        self._editors.clear()
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _build_editors(self, node: NodeModel):
        props = node.properties
        type_id = node.type_id

        # Common: name field (for all nodes that have one)
        if "name" in props:
            edit = QLineEdit(str(props.get("name", "")))
            edit.setPlaceholderText("Enter name…")
            edit.textChanged.connect(lambda val: self._set_prop("name", val))
            self._form_layout.addRow("Name:", edit)
            self._editors["name"] = edit

        # ── Hard-coded utility nodes ────────────────────────────────────
        if type_id == "ParameterVector":
            spin = QSpinBox()
            spin.setRange(1, 10000)
            spin.setValue(int(props.get("size", 1)))
            spin.valueChanged.connect(lambda val: self._set_prop("size", val))
            self._form_layout.addRow("Size:", spin)
            self._editors["size"] = spin

        elif type_id == "Array":
            self._add_data_editors(props)

        elif type_id == "Coordinates":
            self._add_coordinates_editors(props)

        elif type_id == "FieldRequest":
            pass

        elif type_id == "DiagnosticLikelihood":
            pass

        # ── Auto-generated nodes: build editors from default_properties ─
        else:
            self._build_auto_editors(node)

    def _add_data_editors(self, props: dict):
        arr = ArrayEditor("Data values")
        config = props.get("values_config")
        if config:
            arr.set_config(config)
        arr.value_changed.connect(lambda: self._on_array_changed("values", arr))
        self._form_layout.addRow(arr)
        self._editors["values"] = arr

    def _add_field_model_editors(self, props: dict):
        field_name_edit = QLineEdit(str(props.get("field_name", "")))
        field_name_edit.setPlaceholderText("e.g. electron_temperature")
        field_name_edit.textChanged.connect(lambda val: self._set_prop("field_name", val))
        self._form_layout.addRow("Field name:", field_name_edit)
        self._editors["field_name"] = field_name_edit

        axis_name_edit = QComboBox()
        axis_name_edit.setEditable(True)
        axis_name_edit.addItems(["psi", "R", "z", "rho"])
        current = props.get("axis_name", "psi")
        idx = axis_name_edit.findText(current)
        if idx >= 0:
            axis_name_edit.setCurrentIndex(idx)
        else:
            axis_name_edit.setEditText(current)
        axis_name_edit.currentTextChanged.connect(lambda val: self._set_prop("axis_name", val))
        self._form_layout.addRow("Axis name:", axis_name_edit)
        self._editors["axis_name"] = axis_name_edit

    def _add_coordinates_editors(self, props: dict):
        from PySide6.QtWidgets import QHBoxLayout, QListWidget, QListWidgetItem

        coord_names = props.get("coordinate_names", [])

        group = QWidget()
        v = QVBoxLayout(group)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(4)

        add_row = QWidget()
        h = QHBoxLayout(add_row)
        h.setContentsMargins(0, 0, 0, 0)
        new_name_edit = QLineEdit()
        new_name_edit.setPlaceholderText("e.g. phi")
        h.addWidget(new_name_edit)
        add_btn = QPushButton("Add")
        h.addWidget(add_btn)
        remove_btn = QPushButton("Remove")
        h.addWidget(remove_btn)
        v.addWidget(add_row)

        list_widget = QListWidget()
        list_widget.setMaximumHeight(80)
        for name in coord_names:
            list_widget.addItem(QListWidgetItem(name))
        v.addWidget(list_widget)
        self._editors["coord_list"] = list_widget

        self._form_layout.addRow("Coordinates:", group)

        def _sync_names():
            names = [list_widget.item(i).text() for i in range(list_widget.count())]
            self._set_prop("coordinate_names", names)
            if self._node:
                self.ports_changed.emit(self._node.id)

        def _add():
            text = new_name_edit.text().strip()
            if text:
                list_widget.addItem(QListWidgetItem(text))
                new_name_edit.clear()
                _sync_names()

        def _remove():
            row = list_widget.currentRow()
            if row >= 0:
                list_widget.takeItem(row)
                _sync_names()

        add_btn.clicked.connect(_add)
        remove_btn.clicked.connect(_remove)

    def _add_linear_diag_editors(self, props: dict):
        row = QWidget()
        from PySide6.QtWidgets import QHBoxLayout
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        path_edit = QLineEdit(str(props.get("model_matrix_path", "")))
        path_edit.setReadOnly(True)
        path_edit.setPlaceholderText("Select model matrix file…")
        h.addWidget(path_edit)
        btn = QPushButton("Browse…")
        h.addWidget(btn)

        def browse():
            path, _ = QFileDialog.getOpenFileName(
                self, "Import Model Matrix", "",
                "NumPy files (*.npy *.npz);;CSV files (*.csv);;All files (*)",
            )
            if path:
                path_edit.setText(path)
                self._set_prop("model_matrix_path", path)

        btn.clicked.connect(browse)
        self._form_layout.addRow("Model matrix:", row)
        self._editors["model_matrix_path"] = path_edit

    def _add_constant_uncertainty_editors(self, props: dict):
        n_data_spin = QSpinBox()
        n_data_spin.setRange(1, 100000)
        n_data_spin.setValue(int(props.get("n_data", 1)))
        n_data_spin.valueChanged.connect(lambda val: self._set_prop("n_data", val))
        self._form_layout.addRow("n_data:", n_data_spin)
        self._editors["n_data"] = n_data_spin

        param_name_edit = QLineEdit(str(props.get("parameter_name", "")))
        param_name_edit.setPlaceholderText("e.g. sigma")
        param_name_edit.textChanged.connect(lambda val: self._set_prop("parameter_name", val))
        self._form_layout.addRow("Parameter name:", param_name_edit)
        self._editors["parameter_name"] = param_name_edit

    def _add_gaussian_prior_editors(self, props: dict):
        mean_arr = ArrayEditor("Mean")
        mean_arr.set_source(props.get("mean_source", "linspace"))
        mean_arr.value_changed.connect(lambda: self._on_array_changed("mean", mean_arr))
        self._form_layout.addRow(mean_arr)
        self._editors["mean"] = mean_arr

        std_arr = ArrayEditor("Std dev")
        std_arr.set_source(props.get("std_source", "linspace"))
        std_arr.value_changed.connect(lambda: self._on_array_changed("std", std_arr))
        self._form_layout.addRow(std_arr)
        self._editors["std"] = std_arr

    def _build_auto_editors(self, node: NodeModel):
        """Build property editors dynamically from default_properties metadata."""
        from PySide6.QtWidgets import QCheckBox, QHBoxLayout

        props = node.properties
        unresolved_meta = props.get("_unresolved", {})

        for key, default_val in node.spec.default_properties.items():
            if key in ("name", "_class", "_unresolved"):
                continue

            if key in unresolved_meta:
                self._add_unresolved_editor(key, props, unresolved_meta[key])
            elif isinstance(default_val, str):
                edit = QLineEdit(str(props.get(key, default_val)))
                edit.setPlaceholderText(f"Enter {key}…")
                k = key  # capture for closure
                edit.textChanged.connect(lambda val, k=k: self._set_prop(k, val))
                self._form_layout.addRow(f"{key}:", edit)
                self._editors[key] = edit
            elif isinstance(default_val, int):
                spin = QSpinBox()
                spin.setRange(-100000, 100000)
                spin.setValue(int(props.get(key, default_val)))
                k = key
                spin.valueChanged.connect(lambda val, k=k: self._set_prop(k, val))
                self._form_layout.addRow(f"{key}:", spin)
                self._editors[key] = spin
            elif isinstance(default_val, float):
                spin = QDoubleSpinBox()
                spin.setRange(-1e12, 1e12)
                spin.setDecimals(6)
                spin.setValue(float(props.get(key, default_val)))
                k = key
                spin.valueChanged.connect(lambda val, k=k: self._set_prop(k, val))
                self._form_layout.addRow(f"{key}:", spin)
                self._editors[key] = spin
            elif isinstance(default_val, tuple) and len(default_val) == 2:
                row = QWidget()
                h = QHBoxLayout(row)
                h.setContentsMargins(0, 0, 0, 0)
                current = props.get(key, default_val)
                lo = QDoubleSpinBox()
                lo.setRange(-1e12, 1e12)
                lo.setDecimals(6)
                lo.setValue(float(current[0]))
                hi = QDoubleSpinBox()
                hi.setRange(-1e12, 1e12)
                hi.setDecimals(6)
                hi.setValue(float(current[1]))
                h.addWidget(lo)
                h.addWidget(QLabel("to"))
                h.addWidget(hi)
                k = key

                def _on_range_changed(_, k=k, lo=lo, hi=hi):
                    self._set_prop(k, (lo.value(), hi.value()))

                lo.valueChanged.connect(_on_range_changed)
                hi.valueChanged.connect(_on_range_changed)
                self._form_layout.addRow(f"{key}:", row)
                self._editors[key] = row

    def _add_unresolved_editor(self, key: str, props: dict, meta: dict):
        """Add a variable-name field + optional 'Use default' checkbox for unresolved types."""
        from PySide6.QtWidgets import QCheckBox, QHBoxLayout

        type_name = meta.get("type_name", "?")
        has_default = meta.get("has_default", False)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)

        edit = QLineEdit(str(props.get(key, "")))
        edit.setPlaceholderText(f"variable name ({type_name})")
        k = key
        edit.textChanged.connect(lambda val, k=k: self._set_prop(k, val))
        h.addWidget(edit)

        if has_default:
            cb = QCheckBox("Use default")
            use_default_key = f"_use_default_{key}"
            cb.setChecked(props.get(use_default_key, True))
            edit.setEnabled(not cb.isChecked())

            def _on_toggle(checked, k=k, udk=use_default_key, e=edit):
                self._set_prop(udk, checked)
                e.setEnabled(not checked)

            cb.toggled.connect(_on_toggle)
            h.addWidget(cb)
            # Initialize the use-default flag
            if use_default_key not in props:
                props[use_default_key] = True

        self._form_layout.addRow(f"{key}:", row)
        self._editors[key] = edit

    def _set_prop(self, key: str, value):
        if self._node:
            self._node.properties[key] = value
            self.node_updated.emit(self._node.id)

    def _on_array_changed(self, key: str, editor: ArrayEditor):
        if self._node:
            self._node.properties[key + "_config"] = editor.get_config()
            if editor.values is not None:
                self._node.properties[key + "_values"] = editor.values
            self.node_updated.emit(self._node.id)
