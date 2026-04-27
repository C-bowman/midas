from __future__ import annotations

from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsItem,
    QGraphicsRectItem, QGraphicsEllipseItem, QGraphicsPathItem,
    QGraphicsTextItem, QGraphicsDropShadowEffect, QMenu,
)
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPainterPath, QFont, QPainter,
    QWheelEvent, QMouseEvent, QDragEnterEvent, QDropEvent,
    QKeyEvent,
)

from midas_gui.session import (
    GraphModel, NodeModel, Edge, PortSpec, PortDirection, PortType,
    NODE_TYPES,
)
from midas_gui.theme import THEME, CATEGORY_COLORS

PORT_RADIUS = 6
NODE_WIDTH = 180
PORT_SPACING = 36
PORT_MARGIN_TOP = 10


class PortItem(QGraphicsEllipseItem):
    """A single input/output port circle on a node."""

    VALID_COLOR = QColor("#C3E88D")      # green highlight for compatible ports
    DIMMED_OPACITY = 0.12                 # near-invisible for incompatible ports

    def __init__(self, spec: PortSpec, node_item: NodeItem, index: int):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS * 2, PORT_RADIUS * 2)
        self.spec = spec
        self.node_item = node_item
        self.index = index
        self._connected = False
        self._label: QGraphicsTextItem | None = None        # set by NodeItem after creation
        self._type_label: QGraphicsTextItem | None = None   # set by NodeItem after creation

        self.setAcceptHoverEvents(True)
        self.setZValue(3)
        self._update_style()

    def shape(self):
        """Return a larger hit area than the visible circle for easier clicking."""
        path = QPainterPath()
        hit_radius = PORT_RADIUS + 6
        path.addEllipse(-hit_radius, -hit_radius, hit_radius * 2, hit_radius * 2)
        return path

    @property
    def connected(self) -> bool:
        return self._connected

    @connected.setter
    def connected(self, value: bool):
        self._connected = value
        self._update_style()

    def _update_style(self):
        color = QColor(THEME.accent_secondary)
        pen = QPen(color, 1.5)
        self.setPen(pen)
        if self._connected:
            self.setBrush(QBrush(color))
        else:
            self.setBrush(QBrush(QColor(THEME.bg_elevated)))

    def center_scene_pos(self) -> QPointF:
        return self.scenePos()

    def hoverEnterEvent(self, event):
        self.setBrush(QBrush(QColor(THEME.accent_primary)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self._update_style()
        super().hoverLeaveEvent(event)

    # ── Drag-feedback states ───────────────────────────────

    def set_drag_valid(self):
        """Highlight this port as a valid drop target (green glow)."""
        self.setPen(QPen(self.VALID_COLOR, 2.0))
        self.setBrush(QBrush(self.VALID_COLOR))
        self.setOpacity(1.0)
        if self._label:
            self._label.setOpacity(1.0)
        if self._type_label:
            self._type_label.setOpacity(1.0)

    def set_drag_invalid(self):
        """Dim this port to indicate it is not a valid target."""
        self.setOpacity(self.DIMMED_OPACITY)
        if self._label:
            self._label.setOpacity(self.DIMMED_OPACITY)
        if self._type_label:
            self._type_label.setOpacity(self.DIMMED_OPACITY)

    def restore_drag_state(self):
        """Return to normal appearance after a drag ends."""
        self.setOpacity(1.0)
        self._update_style()
        if self._label:
            self._label.setOpacity(1.0)
        if self._type_label:
            self._type_label.setOpacity(1.0)


class NodeItem(QGraphicsRectItem):
    """Visual representation of a single node on the canvas."""

    PORT_FONT = QFont("Segoe UI", 9)
    PORT_TYPE_FONT = QFont("Consolas", 7)
    TITLE_FONT = QFont("Segoe UI", 9, weight=QFont.Weight.DemiBold)
    NAME_FONT = QFont("Segoe UI", 8)
    NODE_TYPE_FONT = QFont("Consolas", 8)
    PORT_PAD = PORT_RADIUS + 6          # space from edge to label text
    LABEL_GAP = 16                      # minimum gap between input and output labels
    TYPE_HEADER_H = 24                  # coloured bar with variable name
    NAME_ROW_H = 20                     # subtitle row with class type
    PORT_LINE_GAP = 1                   # vertical gap between name and type lines

    def __init__(self, node_model: NodeModel, canvas: NodeCanvas):
        self.node_model = node_model
        self.canvas = canvas
        spec = node_model.spec

        node_width, total_height = self._compute_layout()

        super().__init__(0, 0, node_width, total_height)
        self._node_width = node_width

        self.setPos(node_model.x, node_model.y)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setZValue(1)

        # Styling
        category_color = CATEGORY_COLORS.get(spec.category, THEME.accent_secondary)
        self.setPen(QPen(QColor(THEME.border), 1))
        self.setBrush(QBrush(QColor(THEME.bg_elevated)))
        self.setAcceptHoverEvents(True)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(16)
        shadow.setColor(QColor(0, 0, 0, 80))
        shadow.setOffset(2, 2)
        self.setGraphicsEffect(shadow)

        # ── Header: coloured bar with variable name ──────────────
        from PySide6.QtGui import QFontMetricsF
        title_fm = QFontMetricsF(self.TITLE_FONT)
        node_type_fm = QFontMetricsF(self.NODE_TYPE_FONT)

        self._header = QGraphicsRectItem(0, 0, node_width, self.TYPE_HEADER_H, self)
        self._header.setPen(QPen(Qt.PenStyle.NoPen))
        self._header.setBrush(QBrush(QColor(category_color)))
        self._header.setZValue(0)

        var_name = node_model.properties.get("name") or ""
        self._name_label = QGraphicsTextItem(var_name, self)
        self._name_label.setDefaultTextColor(QColor(THEME.bg_base))
        self._name_label.setFont(self.TITLE_FONT)
        self._name_label.document().setDocumentMargin(0)
        name_text_h = title_fm.height()
        self._name_label.setPos(8, (self.TYPE_HEADER_H - name_text_h) / 2)
        self._name_label.setZValue(1)

        # ── Subtitle row: class type ──────────────────────────
        self._type_bg = QGraphicsRectItem(0, self.TYPE_HEADER_H, node_width, self.NAME_ROW_H, self)
        self._type_bg.setPen(QPen(Qt.PenStyle.NoPen))
        type_bg_color = QColor(category_color)
        type_bg_color.setAlphaF(0.15)
        self._type_bg.setBrush(QBrush(type_bg_color))
        self._type_bg.setZValue(0)

        self._type_label = QGraphicsTextItem(spec.display_name, self)
        self._type_label.setDefaultTextColor(QColor(THEME.text_secondary))
        self._type_label.setFont(self.NODE_TYPE_FONT)
        self._type_label.document().setDocumentMargin(0)
        type_text_h = node_type_fm.height()
        self._type_label.setPos(8, self.TYPE_HEADER_H + (self.NAME_ROW_H - type_text_h) / 2)
        self._type_label.setZValue(1)

        # ── Create ports ───────────────────────────────────────
        self.input_ports: dict[str, PortItem] = {}
        self.output_ports: dict[str, PortItem] = {}
        self._port_labels: list[QGraphicsTextItem] = []
        self._hint_label: QGraphicsTextItem | None = None
        self._endpoint_bar = None
        self._create_port_items(node_width, total_height)

    def _compute_layout(self) -> tuple[float, float]:
        """Compute and return (node_width, total_height) based on current ports."""
        from PySide6.QtGui import QFontMetricsF

        spec = self.node_model.spec
        input_specs = self.node_model.effective_input_ports
        n_inputs = len(input_specs)
        n_outputs = len(spec.output_ports)
        n_ports = max(n_inputs, n_outputs, 1)
        full_header = self.TYPE_HEADER_H + self.NAME_ROW_H
        body_height = PORT_MARGIN_TOP + n_ports * PORT_SPACING + 8
        total_height = full_header + body_height

        fm = QFontMetricsF(self.PORT_FONT)
        type_fm = QFontMetricsF(self.PORT_TYPE_FONT)
        title_fm = QFontMetricsF(self.TITLE_FONT)
        name_fm = QFontMetricsF(self.NAME_FONT)
        node_type_fm = QFontMetricsF(self.NODE_TYPE_FONT)

        header_type_width = max(
            title_fm.horizontalAdvance(spec.display_name) + 24,
            node_type_fm.horizontalAdvance(spec.display_name) + 24,
        )
        var_name = self.node_model.properties.get("name") or ""
        name_label_width = name_fm.horizontalAdvance(var_name) + 24

        def _port_col_width(port_spec):
            nw = fm.horizontalAdvance(port_spec.name)
            tw = type_fm.horizontalAdvance(port_spec.type_label)
            return max(nw, tw)

        max_row_width = 0.0
        for i in range(n_ports):
            in_w = _port_col_width(input_specs[i]) if i < n_inputs else 0.0
            out_w = _port_col_width(spec.output_ports[i]) if i < n_outputs else 0.0
            row_width = self.PORT_PAD + in_w + self.LABEL_GAP + out_w + self.PORT_PAD
            max_row_width = max(max_row_width, row_width)

        node_width = max(NODE_WIDTH, max_row_width, header_type_width, name_label_width)
        return node_width, total_height

    def _create_port_items(self, node_width: float, total_height: float):
        """Create port circles, labels, hint label, and endpoint bar."""
        from PySide6.QtGui import QFontMetricsF

        spec = self.node_model.spec
        input_specs = self.node_model.effective_input_ports
        output_specs = spec.output_ports
        full_header = self.TYPE_HEADER_H + self.NAME_ROW_H

        fm = QFontMetricsF(self.PORT_FONT)
        type_fm = QFontMetricsF(self.PORT_TYPE_FONT)
        port_name_h = fm.height()
        port_type_h = type_fm.height()
        stack_h = port_name_h + self.PORT_LINE_GAP + port_type_h

        for i, port_spec in enumerate(input_specs):
            port = PortItem(port_spec, self, i)
            port.setParentItem(self)
            y = full_header + PORT_MARGIN_TOP + i * PORT_SPACING + PORT_SPACING / 2
            port.setPos(0, y)
            self.input_ports[port_spec.name] = port

            top_y = y - stack_h / 2

            label = QGraphicsTextItem(port_spec.name, self)
            label.setDefaultTextColor(QColor(THEME.text_primary))
            label.setFont(self.PORT_FONT)
            label.document().setDocumentMargin(0)
            label.setPos(self.PORT_PAD, top_y)
            label.setZValue(2)
            self._port_labels.append(label)
            port._label = label

            type_lbl = QGraphicsTextItem(port_spec.type_label, self)
            type_lbl.setDefaultTextColor(QColor(THEME.text_secondary))
            type_lbl.setFont(self.PORT_TYPE_FONT)
            type_lbl.document().setDocumentMargin(0)
            type_lbl.setPos(self.PORT_PAD, top_y + port_name_h + self.PORT_LINE_GAP)
            type_lbl.setZValue(2)
            self._port_labels.append(type_lbl)
            port._type_label = type_lbl

        for i, port_spec in enumerate(output_specs):
            port = PortItem(port_spec, self, i)
            port.setParentItem(self)
            y = full_header + PORT_MARGIN_TOP + i * PORT_SPACING + PORT_SPACING / 2
            port.setPos(node_width, y)
            self.output_ports[port_spec.name] = port

            top_y = y - stack_h / 2

            label = QGraphicsTextItem(port_spec.name, self)
            label.setDefaultTextColor(QColor(THEME.text_primary))
            label.setFont(self.PORT_FONT)
            label.document().setDocumentMargin(0)
            label.setZValue(2)
            nw = fm.horizontalAdvance(port_spec.name)
            label.setPos(node_width - self.PORT_PAD - nw, top_y)
            self._port_labels.append(label)
            port._label = label

            type_lbl = QGraphicsTextItem(port_spec.type_label, self)
            type_lbl.setDefaultTextColor(QColor(THEME.text_secondary))
            type_lbl.setFont(self.PORT_TYPE_FONT)
            type_lbl.document().setDocumentMargin(0)
            type_lbl.setZValue(2)
            tw = type_fm.horizontalAdvance(port_spec.type_label)
            type_lbl.setPos(node_width - self.PORT_PAD - tw,
                            top_y + port_name_h + self.PORT_LINE_GAP)
            self._port_labels.append(type_lbl)
            port._type_label = type_lbl

        # Hint label for empty Coordinates nodes
        if self.node_model.type_id == "Coordinates" and not input_specs:
            hint = QGraphicsTextItem("Add coordinates in\nProperties to create ports", self)
            hint.setDefaultTextColor(QColor(THEME.text_secondary))
            hint.setFont(QFont("Segoe UI", 8))
            hint.document().setDocumentMargin(0)
            hint.setPos(8, full_header + PORT_MARGIN_TOP)
            hint.setZValue(2)
            self._hint_label = hint
            for port in self.output_ports.values():
                port.setVisible(False)
                if port._label:
                    port._label.setVisible(False)
                if port._type_label:
                    port._type_label.setVisible(False)

        # Endpoint bar
        if not output_specs:
            category_color = CATEGORY_COLORS.get(spec.category, THEME.accent_secondary)
            bar_width = 4
            bar = QGraphicsRectItem(
                node_width - bar_width, full_header,
                bar_width, total_height - full_header, self
            )
            bar.setPen(QPen(Qt.PenStyle.NoPen))
            bar_color = QColor(category_color)
            bar_color.setAlphaF(0.6)
            bar.setBrush(QBrush(bar_color))
            bar.setZValue(1)
            self._endpoint_bar = bar

    def update_title(self):
        var_name = self.node_model.properties.get("name") or ""
        self._name_label.setPlainText(var_name)

    def rebuild_ports(self):
        """Tear down and recreate all port graphics from effective_input_ports."""
        # Remove old port items and their labels
        for port in list(self.input_ports.values()):
            if port._label:
                self.scene().removeItem(port._label)
            if port._type_label:
                self.scene().removeItem(port._type_label)
            self.scene().removeItem(port)
        for port in list(self.output_ports.values()):
            if port._label:
                self.scene().removeItem(port._label)
            if port._type_label:
                self.scene().removeItem(port._type_label)
            self.scene().removeItem(port)
        self.input_ports.clear()
        self.output_ports.clear()
        self._port_labels.clear()

        # Remove hint label if present
        if self._hint_label:
            self.scene().removeItem(self._hint_label)
            self._hint_label = None

        # Remove endpoint bar if present
        if self._endpoint_bar:
            self.scene().removeItem(self._endpoint_bar)
            self._endpoint_bar = None

        # Recompute layout and resize
        node_width, total_height = self._compute_layout()
        self._node_width = node_width
        self.setRect(0, 0, node_width, total_height)
        self._header.setRect(0, 0, node_width, self.TYPE_HEADER_H)
        self._type_bg.setRect(0, self.TYPE_HEADER_H, node_width, self.NAME_ROW_H)

        # Recreate ports
        self._create_port_items(node_width, total_height)

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.node_model.x = self.pos().x()
            self.node_model.y = self.pos().y()
            self.canvas.update_wires_for_node(self)
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if value:
                self.setPen(QPen(QColor(THEME.selected_border), 2))
            else:
                self.setPen(QPen(QColor(THEME.border), 1))
            self.canvas.node_selection_changed.emit()
        return super().itemChange(change, value)


class WireItem(QGraphicsPathItem):
    """A bezier-curve wire connecting two ports."""

    def __init__(self, edge: Edge, source_port: PortItem, target_port: PortItem):
        super().__init__()
        self.edge = edge
        self.source_port = source_port
        self.target_port = target_port

        color = QColor(THEME.wire_color)
        color.setAlphaF(THEME.wire_opacity)
        self.setPen(QPen(color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(0)

        self.update_path()

    def update_path(self):
        p1 = self.source_port.center_scene_pos()
        p2 = self.target_port.center_scene_pos()
        path = QPainterPath(p1)
        dx = abs(p2.x() - p1.x()) * 0.5
        dx = max(dx, 50)
        path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        self.setPath(path)


class TempWireItem(QGraphicsPathItem):
    """A temporary wire shown while the user is dragging from a port."""

    def __init__(self, start_pos: QPointF, from_output: bool):
        super().__init__()
        self.start_pos = start_pos
        self._from_output = from_output
        color = QColor(THEME.accent_primary)
        color.setAlphaF(0.6)
        self.setPen(QPen(color, 2, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap))
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(10)

    def update_end(self, end_pos: QPointF):
        p1 = self.start_pos
        p2 = end_pos
        path = QPainterPath(p1)
        dx = abs(p2.x() - p1.x()) * 0.5
        dx = max(dx, 50)
        if self._from_output:
            # Output port is on the right side of the node — curve goes right then left
            path.cubicTo(p1.x() + dx, p1.y(), p2.x() - dx, p2.y(), p2.x(), p2.y())
        else:
            # Input port is on the left side of the node — curve goes left then right
            path.cubicTo(p1.x() - dx, p1.y(), p2.x() + dx, p2.y(), p2.x(), p2.y())
        self.setPath(path)


class NodeCanvas(QGraphicsView):
    """The main node-graph canvas widget."""

    node_selection_changed = Signal()
    graph_modified = Signal()

    def __init__(self, graph: GraphModel, parent=None):
        self._scene = QGraphicsScene(parent)
        super().__init__(self._scene, parent)
        self.graph = graph

        self.node_items: dict[str, NodeItem] = {}
        self.wire_items: list[WireItem] = []
        self._temp_wire: TempWireItem | None = None
        self._drag_source_port: PortItem | None = None
        self._zoom = 1.0

        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(THEME.bg_base)))
        self.setSceneRect(-5000, -5000, 10000, 10000)
        self.setAcceptDrops(True)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        grid_size = 25
        color = QColor(THEME.canvas_grid)
        painter.setPen(QPen(color, 1.5))

        left = int(rect.left()) - (int(rect.left()) % grid_size)
        top = int(rect.top()) - (int(rect.top()) % grid_size)

        x = left
        while x < rect.right():
            y = top
            while y < rect.bottom():
                painter.drawPoint(QPointF(x, y))
                y += grid_size
            x += grid_size

    # ── Node management ────────────────────────────────────────

    def add_node(self, type_id: str, x: float = 0.0, y: float = 0.0) -> NodeItem:
        node_model = self.graph.add_node(type_id, x, y)
        item = self._create_node_item(node_model)
        self.graph_modified.emit()
        return item

    def _create_node_item(self, node_model: NodeModel) -> NodeItem:
        """Create the visual item for an existing NodeModel."""
        item = NodeItem(node_model, self)
        self._scene.addItem(item)
        self.node_items[node_model.id] = item
        return item

    def remove_selected_nodes(self):
        to_remove = [item for item in self.node_items.values() if item.isSelected()]
        for item in to_remove:
            self._remove_node_item(item)

    def _remove_node_item(self, item: NodeItem):
        node_id = item.node_model.id
        wires_to_remove = [w for w in self.wire_items
                           if w.edge.source_node_id == node_id or w.edge.target_node_id == node_id]
        for w in wires_to_remove:
            self._remove_wire(w)
        # Release any mouse grab and clear selection before removal
        # to prevent "cannot ungrab mouse without scene" warnings.
        item.setSelected(False)
        if item.scene():
            item.ungrabMouse()
        self._scene.removeItem(item)
        self.node_items.pop(node_id, None)
        self.graph.remove_node(node_id)
        self.node_selection_changed.emit()
        self.graph_modified.emit()

    # ── Wire management ────────────────────────────────────────

    def add_wire(self, edge: Edge) -> WireItem | None:
        src_item = self.node_items.get(edge.source_node_id)
        tgt_item = self.node_items.get(edge.target_node_id)
        if not src_item or not tgt_item:
            return None

        src_port = src_item.output_ports.get(edge.source_port_name)
        tgt_port = tgt_item.input_ports.get(edge.target_port_name)
        if not src_port or not tgt_port:
            return None

        wire = WireItem(edge, src_port, tgt_port)
        self._scene.addItem(wire)
        self.wire_items.append(wire)
        src_port.connected = True
        tgt_port.connected = True
        self.graph_modified.emit()
        return wire

    def _remove_wire(self, wire: WireItem):
        self.graph.remove_edge(wire.edge)
        self._scene.removeItem(wire)
        self.wire_items.remove(wire)
        self._update_port_connected_state(wire.source_port)
        self._update_port_connected_state(wire.target_port)
        self.graph_modified.emit()

    def _update_port_connected_state(self, port: PortItem):
        node_id = port.node_item.node_model.id
        if port.spec.direction == PortDirection.OUTPUT:
            port.connected = any(
                w.edge.source_node_id == node_id and w.edge.source_port_name == port.spec.name
                for w in self.wire_items
            )
        else:
            port.connected = any(
                w.edge.target_node_id == node_id and w.edge.target_port_name == port.spec.name
                for w in self.wire_items
            )

    def update_wires_for_node(self, node_item: NodeItem):
        node_id = node_item.node_model.id
        for wire in self.wire_items:
            if wire.edge.source_node_id == node_id or wire.edge.target_node_id == node_id:
                wire.update_path()

    def rebuild_node_ports(self, node_id: str):
        """Rebuild ports for a node after its dynamic port set has changed.

        Removes wires connected to ports that no longer exist, then
        rebuilds the visual port items and reconnects surviving wires.
        """
        node_item = self.node_items.get(node_id)
        if not node_item:
            return

        node_model = node_item.node_model
        # Determine which port names will exist after rebuild
        new_input_names = {p.name for p in node_model.effective_input_ports}
        new_output_names = {p.name for p in node_model.spec.output_ports}

        # Remove wires connected to ports that are being removed
        wires_to_remove = []
        for wire in self.wire_items:
            if wire.edge.target_node_id == node_id and wire.edge.target_port_name not in new_input_names:
                wires_to_remove.append(wire)
            elif wire.edge.source_node_id == node_id and wire.edge.source_port_name not in new_output_names:
                wires_to_remove.append(wire)
        for wire in wires_to_remove:
            self._remove_wire(wire)

        # Rebuild the visual ports
        node_item.rebuild_ports()

        # Reconnect surviving wires to the new port items
        for wire in self.wire_items:
            if wire.edge.source_node_id == node_id:
                new_port = node_item.output_ports.get(wire.edge.source_port_name)
                if new_port:
                    wire.source_port = new_port
                    new_port.connected = True
            if wire.edge.target_node_id == node_id:
                new_port = node_item.input_ports.get(wire.edge.target_port_name)
                if new_port:
                    wire.target_port = new_port
                    new_port.connected = True
            if wire.edge.source_node_id == node_id or wire.edge.target_node_id == node_id:
                wire.update_path()

    # ── Mouse interaction: wiring ──────────────────────────────

    def _find_port_at(self, view_pos) -> PortItem | None:
        """Find a PortItem at the given view position, searching all items."""
        for item in self.items(view_pos):
            if isinstance(item, PortItem):
                return item
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position().toPoint()
            port = self._find_port_at(pos)
            if port:
                self._start_wire_drag(port, event)
                return
            # If clicking on empty canvas (no item), start rubber band selection
            item = self.itemAt(pos)
            if item is None:
                self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
                super().mousePressEvent(event)
                return
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake = QMouseEvent(
                event.type(), event.position(), event.globalPosition(),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mousePressEvent(fake)
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._temp_wire and self._drag_source_port:
            scene_pos = self.mapToScene(event.position().toPoint())
            self._temp_wire.update_end(scene_pos)
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            fake = QMouseEvent(
                event.type(), event.position(), event.globalPosition(),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            super().mouseReleaseEvent(fake)
            return
        if event.button() == Qt.MouseButton.LeftButton and self._temp_wire:
            self._finish_wire_drag(event)
            return
        super().mouseReleaseEvent(event)
        # Reset rubber band drag mode after selection completes
        if self.dragMode() == QGraphicsView.DragMode.RubberBandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def _start_wire_drag(self, port: PortItem, event):
        self._drag_source_port = port
        start = port.center_scene_pos()
        from_output = port.spec.direction == PortDirection.OUTPUT
        self._temp_wire = TempWireItem(start, from_output)
        self._scene.addItem(self._temp_wire)
        self._highlight_compatible_ports(port)

    def _finish_wire_drag(self, event):
        if self._temp_wire:
            self._scene.removeItem(self._temp_wire)
            self._temp_wire = None

        self._restore_all_ports()

        pos = event.position().toPoint()
        target_port = self._find_port_at(pos)
        if target_port and self._drag_source_port:
            src = self._drag_source_port
            tgt = target_port

            # Determine direction: allow dragging from input to output too
            if src.spec.direction == PortDirection.OUTPUT and tgt.spec.direction == PortDirection.INPUT:
                edge = self.graph.add_edge(
                    src.node_item.node_model.id, src.spec.name,
                    tgt.node_item.node_model.id, tgt.spec.name,
                )
            elif src.spec.direction == PortDirection.INPUT and tgt.spec.direction == PortDirection.OUTPUT:
                edge = self.graph.add_edge(
                    tgt.node_item.node_model.id, tgt.spec.name,
                    src.node_item.node_model.id, src.spec.name,
                )
            else:
                edge = None

            if edge:
                self.add_wire(edge)

        self._drag_source_port = None

    # ── Port highlighting during drag ──────────────────────────

    def _highlight_compatible_ports(self, source_port: PortItem):
        """Highlight ports that can accept a connection from *source_port*."""
        src_node_id = source_port.node_item.node_model.id
        src_dir = source_port.spec.direction
        src_types = set(source_port.spec.port_types)

        for node_item in self.node_items.values():
            # Pick the opposite-direction port dict
            if src_dir == PortDirection.OUTPUT:
                candidate_ports = node_item.input_ports
            else:
                candidate_ports = node_item.output_ports

            # Also collect same-direction ports to dim them
            if src_dir == PortDirection.OUTPUT:
                same_dir_ports = node_item.output_ports
            else:
                same_dir_ports = node_item.input_ports

            for port in same_dir_ports.values():
                if port is source_port:
                    continue
                port.set_drag_invalid()

            for port in candidate_ports.values():
                # Same node → invalid
                if node_item.node_model.id == src_node_id:
                    port.set_drag_invalid()
                    continue

                # Type mismatch → no overlapping types
                if not src_types & set(port.spec.port_types):
                    port.set_drag_invalid()
                    continue

                # For input ports, check if already connected
                if port.spec.direction == PortDirection.INPUT:
                    already_connected = any(
                        e.target_node_id == node_item.node_model.id
                        and e.target_port_name == port.spec.name
                        for e in self.graph.edges
                    )
                    if already_connected:
                        port.set_drag_invalid()
                        continue

                port.set_drag_valid()

    def _restore_all_ports(self):
        """Restore every port to its normal appearance."""
        for node_item in self.node_items.values():
            for port in node_item.input_ports.values():
                port.restore_drag_state()
            for port in node_item.output_ports.values():
                port.restore_drag_state()

    # ── Zoom ───────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom *= factor
            self.scale(factor, factor)
        else:
            self._zoom /= factor
            self.scale(1 / factor, 1 / factor)

    # ── Context menu ───────────────────────────────────────────

    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())

        # Find the NodeItem if we clicked on a child
        node_item = None
        check = item
        while check:
            if isinstance(check, NodeItem):
                node_item = check
                break
            check = check.parentItem()

        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {THEME.bg_elevated};
                color: {THEME.text_primary};
                border: 1px solid {THEME.border};
            }}
            QMenu::item:selected {{
                background-color: {THEME.accent_secondary};
                color: {THEME.bg_base};
            }}
        """)

        if node_item:
            delete_action = menu.addAction("Delete Node")
            disconnect_action = menu.addAction("Disconnect All")

            action = menu.exec(event.globalPos())
            if action == delete_action:
                self._remove_node_item(node_item)
            elif action == disconnect_action:
                wires = [w for w in self.wire_items
                         if w.edge.source_node_id == node_item.node_model.id
                         or w.edge.target_node_id == node_item.node_model.id]
                for w in list(wires):
                    self._remove_wire(w)
        else:
            scene_pos = self.mapToScene(event.pos())
            for type_id, spec in NODE_TYPES.items():
                action = menu.addAction(f"Add {spec.display_name}")
                action.setData(type_id)

            action = menu.exec(event.globalPos())
            if action and action.data():
                self.add_node(action.data(), scene_pos.x(), scene_pos.y())

    # ── Drag & drop from palette ───────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        type_id = event.mimeData().text()
        if type_id in NODE_TYPES:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.add_node(type_id, scene_pos.x(), scene_pos.y())
            event.acceptProposedAction()

    def selected_node(self) -> NodeModel | None:
        for item in self.node_items.values():
            if item.isSelected():
                return item.node_model
        return None

    def selected_node_item(self) -> NodeItem | None:
        for item in self.node_items.values():
            if item.isSelected():
                return item
        return None

    # ── Keyboard shortcuts ─────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        modifiers = event.modifiers()

        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.remove_selected_nodes()
            return
        if key == Qt.Key.Key_A and modifiers & Qt.KeyboardModifier.ControlModifier:
            for item in self.node_items.values():
                item.setSelected(True)
            return
        if key == Qt.Key.Key_D and modifiers & Qt.KeyboardModifier.ControlModifier:
            self.duplicate_selected_node()
            return
        if key == Qt.Key.Key_Escape:
            for item in self.node_items.values():
                item.setSelected(False)
            return

        super().keyPressEvent(event)

    def duplicate_selected_node(self):
        """Duplicate the currently selected node, offset slightly."""
        source_item = self.selected_node_item()
        if not source_item:
            return
        source = source_item.node_model
        new_item = self.add_node(
            source.type_id,
            source.x + 30,
            source.y + 30,
        )
        # Copy properties (except auto-generated name)
        import copy
        for key, val in source.properties.items():
            if key == "name":
                continue
            new_item.node_model.properties[key] = copy.deepcopy(val)
        new_item.update_title()
