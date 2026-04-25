from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow, QDockWidget, QMenuBar, QStatusBar,
)
from PySide6.QtCore import Qt

from midas_gui.session import GraphModel
from midas_gui.widgets.node_canvas import NodeCanvas
from midas_gui.widgets.node_palette import NodePalette
from midas_gui.widgets.properties_panel import PropertiesPanel
from midas_gui.widgets.code_preview import CodePreview
from midas_gui.theme import THEME


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDAS — Analysis Builder")
        self.resize(1400, 850)

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
        self.palette_widget = NodePalette(self)
        self.palette_dock.setWidget(self.palette_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.palette_dock)

        # Right dock: properties panel
        self.props_dock = QDockWidget("Properties", self)
        self.props_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.props_panel = PropertiesPanel(self)
        self.props_dock.setWidget(self.props_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.props_dock)

        # Bottom dock: code preview
        self.code_dock = QDockWidget("Generated Script", self)
        self.code_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.TopDockWidgetArea
        )
        self.code_preview = CodePreview(self.graph, self)
        self.code_dock.setWidget(self.code_preview)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.code_dock)

        # Menu bar
        self._build_menu_bar()

        # Status bar
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage("Ready")

        # Connect signals
        self.canvas.node_selection_changed.connect(self._on_selection_changed)
        self.props_panel.node_updated.connect(self._on_node_updated)

    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction("Export Script…", self._export_script)
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
        node_item = self.canvas.node_items.get(node_id)
        if node_item:
            node_item.update_title()
        self.code_preview.refresh()

    def _export_script(self):
        self.code_preview._export()
