from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any
import uuid


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    return s.lower()


class PortType(Enum):
    ARRAY = auto()
    PARAMS = auto()
    COORDINATES = auto()
    FIELD = auto()
    FIELD_REQUEST = auto()
    DIAGNOSTIC_MODEL = auto()
    UNCERTAINTIES = auto()
    LIKELIHOOD = auto()

    @property
    def display(self) -> str:
        """Short label shown on port type annotations."""
        return {
            PortType.ARRAY: "Array",
            PortType.PARAMS: "ParameterVector",
            PortType.COORDINATES: "Coordinates",
            PortType.FIELD: "Field",
            PortType.FIELD_REQUEST: "FieldRequest",
            PortType.DIAGNOSTIC_MODEL: "DiagnosticModel",
            PortType.UNCERTAINTIES: "UncertaintyModel",
            PortType.LIKELIHOOD: "LikelihoodFunction",
        }[self]


class PortDirection(Enum):
    INPUT = auto()
    OUTPUT = auto()


@dataclass
class PortSpec:
    name: str
    port_types: tuple[PortType, ...]
    direction: PortDirection
    required: bool = True

    @property
    def type_label(self) -> str:
        """Human-readable type annotation for the port."""
        return " | ".join(pt.display for pt in self.port_types)


# ── Node type registry ──────────────────────────────────────────────────

@dataclass(frozen=True)
class NodeTypeSpec:
    type_id: str
    display_name: str
    category: str
    input_ports: tuple[PortSpec, ...]
    output_ports: tuple[PortSpec, ...]
    default_properties: dict[str, Any] = field(default_factory=dict)


def _normalize_types(pt) -> tuple[PortType, ...]:
    """Accept a single PortType or a tuple of them and always return a tuple."""
    if isinstance(pt, PortType):
        return (pt,)
    return tuple(pt)


def _make_spec(
    type_id: str,
    display_name: str,
    category: str,
    inputs: list,
    outputs: list,
    defaults: dict[str, Any] | None = None,
) -> NodeTypeSpec:
    input_ports = []
    for item in inputs:
        if len(item) == 3:
            name, pt, req = item
        else:
            name, pt = item
            req = True
        input_ports.append(PortSpec(name, _normalize_types(pt), PortDirection.INPUT, req))

    output_ports = tuple(
        PortSpec(name, _normalize_types(pt), PortDirection.OUTPUT) for name, pt in outputs
    )
    return NodeTypeSpec(
        type_id=type_id,
        display_name=display_name,
        category=category,
        input_ports=tuple(input_ports),
        output_ports=output_ports,
        default_properties=defaults or {},
    )


NODE_TYPES: dict[str, NodeTypeSpec] = {}


def _register(*specs: NodeTypeSpec):
    for spec in specs:
        NODE_TYPES[spec.type_id] = spec


_register(
    # Parameters & Data
    _make_spec(
        "ParameterVector", "ParameterVector", "Parameters & Data",
        [],
        [("parameter_vector", PortType.PARAMS)],
        {"name": "", "size": 1},
    ),
    _make_spec(
        "Array", "Array", "Parameters & Data",
        [],
        [("data", PortType.ARRAY)],
        {"source_type": "file", "file_path": "", "values": None},
    ),
    _make_spec(
        "Coordinates", "Coordinates", "Parameters & Data",
        [],
        [("coordinates", PortType.COORDINATES)],
        {"name": "", "coordinate_names": ["R", "z"]},
    ),
    _make_spec(
        "FieldRequest", "FieldRequest", "Parameters & Data",
        [
            ("field", PortType.FIELD),
            ("coordinates", PortType.COORDINATES),
        ],
        [("field_request", PortType.FIELD_REQUEST)],
        {"name": ""},
    ),
    # Field Models
    _make_spec(
        "PiecewiseLinearField", "PiecewiseLinearField", "Field Models",
        [("axis", PortType.ARRAY)],
        [("field", PortType.FIELD)],
        {"name": "", "field_name": "", "axis_name": "psi"},
    ),
    _make_spec(
        "CubicSplineField", "CubicSplineField", "Field Models",
        [("axis", PortType.ARRAY)],
        [("field", PortType.FIELD)],
        {"name": "", "field_name": "", "axis_name": "psi"},
    ),
    # Diagnostic Models
    _make_spec(
        "LinearDiagnosticModel", "LinearDiagnosticModel", "Diagnostic Models",
        [
            ("field", PortType.FIELD_REQUEST),
            ("model_matrix", PortType.ARRAY),
        ],
        [("diagnostic_model", PortType.DIAGNOSTIC_MODEL)],
        {"name": ""},
    ),
    # Likelihoods
    _make_spec(
        "GaussianLikelihood", "GaussianLikelihood", "Likelihoods",
        [
            ("y_data", PortType.ARRAY),
            ("sigma", (PortType.ARRAY, PortType.UNCERTAINTIES), False),
        ],
        [("likelihood", PortType.LIKELIHOOD)],
        {"name": ""},
    ),
    _make_spec(
        "DiagnosticLikelihood", "DiagnosticLikelihood", "Likelihoods",
        [
            ("diagnostic_model", PortType.DIAGNOSTIC_MODEL),
            ("likelihood", PortType.LIKELIHOOD),
        ],
        [],
        {"name": ""},
    ),
    # Uncertainty Models
    _make_spec(
        "ConstantUncertainty", "ConstantUncertainty", "Uncertainty Models",
        [],
        [("uncertainties", PortType.UNCERTAINTIES)],
        {"name": "", "n_data": 1, "parameter_name": ""},
    ),
    # Priors
    _make_spec(
        "GaussianPrior", "GaussianPrior", "Priors",
        [
            ("mean", PortType.ARRAY),
            ("standard_deviation", PortType.ARRAY),
            ("field_request", PortType.FIELD_REQUEST, False),
            ("parameter_vector", PortType.PARAMS, False),
        ],
        [],
        {"name": ""},
    ),
)


# ── Runtime graph model ─────────────────────────────────────────────────

@dataclass
class NodeModel:
    id: str
    type_id: str
    x: float = 0.0
    y: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> NodeTypeSpec:
        return NODE_TYPES[self.type_id]


@dataclass(frozen=True)
class Edge:
    source_node_id: str
    source_port_name: str
    target_node_id: str
    target_port_name: str


class GraphModel:
    """In-memory representation of the node graph."""

    def __init__(self):
        self.nodes: dict[str, NodeModel] = {}
        self.edges: list[Edge] = []
        self._name_counters: dict[str, int] = {}  # type_id -> next number

    def _next_name(self, type_id: str) -> str:
        """Return an auto-generated snake_case name like 'gaussian_prior_1'."""
        count = self._name_counters.get(type_id, 0) + 1
        self._name_counters[type_id] = count
        return f"{_camel_to_snake(type_id)}_{count}"

    def add_node(self, type_id: str, x: float = 0.0, y: float = 0.0) -> NodeModel:
        spec = NODE_TYPES[type_id]
        node_id = uuid.uuid4().hex[:8]
        props = dict(spec.default_properties)
        props["name"] = self._next_name(type_id)
        node = NodeModel(id=node_id, type_id=type_id, x=x, y=y, properties=props)
        self.nodes[node_id] = node
        return node

    def remove_node(self, node_id: str):
        self.edges = [
            e for e in self.edges
            if e.source_node_id != node_id and e.target_node_id != node_id
        ]
        self.nodes.pop(node_id, None)

    def can_connect(
        self,
        source_node_id: str,
        source_port_name: str,
        target_node_id: str,
        target_port_name: str,
    ) -> bool:
        if source_node_id == target_node_id:
            return False

        src_node = self.nodes.get(source_node_id)
        tgt_node = self.nodes.get(target_node_id)
        if not src_node or not tgt_node:
            return False

        src_port = self._find_port(src_node.spec.output_ports, source_port_name)
        tgt_port = self._find_port(tgt_node.spec.input_ports, target_port_name)
        if not src_port or not tgt_port:
            return False

        # Check type compatibility: the output's types must overlap with the input's accepted types
        if not set(src_port.port_types) & set(tgt_port.port_types):
            return False

        # Check if target port already has a connection
        for e in self.edges:
            if e.target_node_id == target_node_id and e.target_port_name == target_port_name:
                return False

        return True

    def add_edge(
        self,
        source_node_id: str,
        source_port_name: str,
        target_node_id: str,
        target_port_name: str,
    ) -> Edge | None:
        if not self.can_connect(source_node_id, source_port_name, target_node_id, target_port_name):
            return None
        edge = Edge(source_node_id, source_port_name, target_node_id, target_port_name)
        self.edges.append(edge)
        return edge

    def remove_edge(self, edge: Edge):
        self.edges = [e for e in self.edges if e is not edge]

    def edges_for_node(self, node_id: str) -> list[Edge]:
        return [e for e in self.edges if e.source_node_id == node_id or e.target_node_id == node_id]

    @staticmethod
    def _find_port(ports: tuple[PortSpec, ...], name: str) -> PortSpec | None:
        for p in ports:
            if p.name == name:
                return p
        return None

    def validate(self) -> list[str]:
        """Return a list of validation error messages."""
        errors = []
        for node in self.nodes.values():
            for port in node.spec.input_ports:
                if not port.required:
                    continue
                connected = any(
                    e.target_node_id == node.id and e.target_port_name == port.name
                    for e in self.edges
                )
                if not connected:
                    errors.append(
                        f"Node '{node.properties.get('name', node.type_id)}' ({node.id}): "
                        f"required input '{port.name}' is not connected."
                    )
        return errors
