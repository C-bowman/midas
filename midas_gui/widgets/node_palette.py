from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QLabel,
)
from PySide6.QtCore import Qt, QMimeData
from PySide6.QtGui import QDrag, QColor

from midas_gui.session import NODE_TYPES
from midas_gui.theme import THEME, CATEGORY_COLORS


class NodePalette(QWidget):
    """Left sidebar listing draggable node types grouped by category."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("Node Palette")
        header.setStyleSheet(
            f"font-weight: bold; font-size: 12px; color: {THEME.text_primary}; padding: 4px;"
        )
        layout.addWidget(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setDragEnabled(True)
        self.tree.setIndentation(16)
        self.tree.setRootIsDecorated(True)
        layout.addWidget(self.tree)

        self._build_tree()
        self.tree.expandAll()

        self.tree.startDrag = self._start_drag

    def _build_tree(self):
        categories: dict[str, QTreeWidgetItem] = {}

        for type_id, spec in NODE_TYPES.items():
            cat = spec.category
            if cat not in categories:
                cat_item = QTreeWidgetItem([cat])
                cat_item.setFlags(
                    cat_item.flags() & ~Qt.ItemFlag.ItemIsDragEnabled
                )
                color = CATEGORY_COLORS.get(cat, THEME.text_primary)
                cat_item.setForeground(0, QColor(color))
                self.tree.addTopLevelItem(cat_item)
                categories[cat] = cat_item

            node_item = QTreeWidgetItem([spec.display_name])
            node_item.setData(0, Qt.ItemDataRole.UserRole, type_id)
            categories[cat].addChild(node_item)

    def refresh(self):
        """Rebuild the tree to reflect newly registered node types."""
        self.tree.clear()
        self._build_tree()
        self.tree.expandAll()

    def _start_drag(self, supported_actions):
        item = self.tree.currentItem()
        if not item:
            return
        type_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not type_id:
            return

        drag = QDrag(self.tree)
        mime = QMimeData()
        mime.setText(type_id)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)
