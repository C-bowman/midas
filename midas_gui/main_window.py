from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QMenuBar, QStatusBar, QFileDialog, QMessageBox,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QAction, QCloseEvent

from midas_gui.session import GraphModel, NODE_TYPES
from midas_gui.settings import Settings, SettingsDialog
from midas_gui.widgets.node_canvas import NodeCanvas
from midas_gui.widgets.node_palette import NodePalette
from midas_gui.widgets.properties_panel import PropertiesPanel
from midas_gui.widgets.code_preview import CodePreview
from midas_gui.theme import THEME


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings):
        super().__init__()
        self.resize(1400, 850)
        self._settings = settings

        self._current_file: str | None = None
        self._imported_modules: list[str] = []
        self._dirty = False

        # Data model
        self.graph = GraphModel()

        # Central widget: node canvas
        self.canvas = NodeCanvas(self.graph, self)
        self.setCentralWidget(self.canvas)

        # Left dock: node palette
        self.palette_dock = QDockWidget("Node Palette", self)
        self.palette_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.palette_widget = NodePalette(settings, self)
        self.palette_dock.setWidget(self.palette_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.palette_dock)

        # Right dock: properties panel
        self.props_dock = QDockWidget("Properties", self)
        self.props_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.props_panel = PropertiesPanel(settings, self)
        self.props_dock.setWidget(self.props_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)

        # Bottom dock: code preview
        self.code_dock = QDockWidget("Generated Script", self)
        self.code_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.code_preview = CodePreview(self.graph, settings, self)
        self.code_dock.setWidget(self.code_preview)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.code_dock)

        # Menu bar
        self._build_menu_bar()

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # Connect signals
        self.canvas.node_selection_changed.connect(self._on_selection_changed)
        self.canvas.graph_modified.connect(self._on_graph_modified)
        self.props_panel.node_updated.connect(self._on_node_updated)
        self.props_panel.ports_changed.connect(self._on_ports_changed)

        self._update_title()

    def _update_title(self):
        name = "Untitled"
        if self._current_file:
            from pathlib import Path
            name = Path(self._current_file).stem
        self.setWindowTitle(f"{name} — MIDAS Analysis Builder")

    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")

        open_action = file_menu.addAction("&Open…", self._open_file)
        open_action.setShortcut(QKeySequence.StandardKey.Open)

        save_action = file_menu.addAction("&Save", self._save_file)
        save_action.setShortcut(QKeySequence.StandardKey.Save)

        file_menu.addAction("Save &As…", self._save_file_as)

        file_menu.addSeparator()
        file_menu.addAction("Import Module…", self._import_module)
        file_menu.addAction("Export Script…", self._export_script)
        file_menu.addSeparator()
        file_menu.addAction("Settings…", self._open_settings)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        # View menu
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.palette_dock.toggleViewAction())
        view_menu.addAction(self.props_dock.toggleViewAction())
        view_menu.addAction(self.code_dock.toggleViewAction())

    def _on_selection_changed(self):
        node = self.canvas.selected_node()
        self.props_panel.set_node(node)
        if node:
            self.statusBar().showMessage(f"Selected: {node.spec.display_name} ({node.id})")
        else:
            self.statusBar().showMessage("Ready")
        self.code_preview.refresh()

    def _on_node_updated(self, node_id: str):
        self._dirty = True
        node_item = self.canvas.node_items.get(node_id)
        if node_item:
            node_item.update_title()
        self.code_preview.refresh()

    def _on_ports_changed(self, node_id: str):
        self._dirty = True
        self.canvas.rebuild_node_ports(node_id)
        self.code_preview.refresh()

    def _on_graph_modified(self):
        self._dirty = True
        self.code_preview.refresh()

    def closeEvent(self, event: QCloseEvent):
        if self._dirty and self.graph.nodes:
            reply = QMessageBox.question(
                self, "Unsaved Changes",
                "You have unsaved changes. Are you sure you want to exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
        event.accept()

    # ── Save / Load ────────────────────────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Session", "",
            "MIDAS sessions (*.midas);;All files (*)",
        )
        if not path:
            return

        from midas_gui.session_io import load_session, read_imported_modules
        from midas_gui.introspection import discover_user_module

        # Re-import user modules BEFORE loading nodes so their types are registered
        try:
            imported_modules = read_imported_modules(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Could not read session:\n{exc}")
            return

        for mod_path in imported_modules:
            specs = discover_user_module(mod_path)
            for type_id, spec in specs.items():
                if type_id not in NODE_TYPES:
                    NODE_TYPES[type_id] = spec

        try:
            graph, imported_modules = load_session(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Failed", f"Could not load session:\n{exc}")
            return

        self._load_graph(graph)
        self._imported_modules = list(imported_modules)
        self._current_file = path
        self._dirty = False
        self._update_title()
        self.palette_widget.refresh()
        self.statusBar().showMessage(f"Opened {path}")

    def _save_file(self):
        if self._current_file:
            self._do_save(self._current_file)
        else:
            self._save_file_as()

    def _save_file_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "",
            "MIDAS sessions (*.midas);;All files (*)",
        )
        if not path:
            return
        if not path.endswith(".midas"):
            path += ".midas"
        self._do_save(path)
        self._current_file = path
        self._update_title()

    def _do_save(self, path: str):
        from midas_gui.session_io import save_session

        # Sync node positions from canvas items
        for node_id, item in self.canvas.node_items.items():
            node = self.graph.nodes.get(node_id)
            if node:
                node.x = item.pos().x()
                node.y = item.pos().y()

        try:
            save_session(self.graph, path, self._imported_modules)
            self._dirty = False
            self.statusBar().showMessage(f"Saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save session:\n{exc}")

    def _load_graph(self, graph: GraphModel):
        """Replace the current graph and rebuild the canvas."""
        # Clear existing canvas
        self.props_panel.set_node(None)
        for item in list(self.canvas.node_items.values()):
            self.canvas._remove_node_item(item)

        # Replace graph model
        self.graph = graph
        self.canvas.graph = graph
        self.code_preview.graph = graph

        # Recreate node items
        from midas_gui.widgets.node_canvas import NodeItem
        for node in graph.nodes.values():
            item = NodeItem(node, self.canvas)
            self.canvas._scene.addItem(item)
            self.canvas.node_items[node.id] = item

        # Recreate wire items
        for edge in graph.edges:
            self.canvas.add_wire(edge)

        self.code_preview.refresh()

    def _import_module(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Module", "",
            "Python files (*.py);;All files (*)",
        )
        if not path:
            return

        from midas_gui.introspection import discover_user_module
        from midas_gui.session import NODE_TYPES

        specs = discover_user_module(path)
        if not specs:
            QMessageBox.information(
                self, "Import Module",
                f"No supported node types found in:\n{path}\n\n"
                "The module must contain concrete subclasses of "
                "DiagnosticModel, FieldModel, LikelihoodFunction, "
                "UncertaintyModel, or BasePrior.",
            )
            return

        new_names = []
        for type_id, spec in specs.items():
            if type_id not in NODE_TYPES:
                NODE_TYPES[type_id] = spec
                new_names.append(spec.display_name)

        if new_names:
            self.palette_widget.refresh()
            if path not in self._imported_modules:
                self._imported_modules.append(path)
            self.statusBar().showMessage(
                f"Imported {len(new_names)} node type(s): {', '.join(new_names)}"
            )
        else:
            QMessageBox.information(
                self, "Import Module",
                "All node types in that module are already registered.",
            )

    def _export_script(self):
        self.code_preview._export()

    def _open_settings(self):
        dialog = SettingsDialog(self._settings, self)
        dialog.exec()
