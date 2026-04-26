"""Tests for the session data model (midas_gui.session)."""
from __future__ import annotations

import pytest

from midas_gui.session import (
    GraphModel,
    NodeModel,
    Edge,
    PortSpec,
    PortType,
    PortDirection,
    NODE_TYPES,
    _camel_to_snake,
)


# ── Helpers ────────────────────────────────────────────────────────────


class TestCamelToSnake:
    def test_simple(self):
        assert _camel_to_snake("GaussianPrior") == "gaussian_prior"

    def test_single_word(self):
        assert _camel_to_snake("Array") == "array"

    def test_consecutive_caps(self):
        assert _camel_to_snake("LinearDiagnosticModel") == "linear_diagnostic_model"

    def test_already_snake(self):
        assert _camel_to_snake("some_name") == "some_name"


# ── PortSpec ───────────────────────────────────────────────────────────


class TestPortSpec:
    def test_type_label_single(self):
        ps = PortSpec("x", (PortType.ARRAY,), PortDirection.INPUT)
        assert ps.type_label == "Array"

    def test_type_label_multi(self):
        ps = PortSpec("sigma", (PortType.ARRAY, PortType.UNCERTAINTIES), PortDirection.INPUT)
        assert ps.type_label == "Array | UncertaintyModel"

    def test_required_default_true(self):
        ps = PortSpec("x", (PortType.ARRAY,), PortDirection.INPUT)
        assert ps.required is True


# ── GraphModel: add_node ───────────────────────────────────────────────


class TestAddNode:
    def test_adds_node_to_graph(self):
        g = GraphModel()
        node = g.add_node("Array")
        assert node.id in g.nodes
        assert g.nodes[node.id] is node

    def test_node_has_correct_type(self):
        g = GraphModel()
        node = g.add_node("Array")
        assert node.type_id == "Array"

    def test_node_has_position(self):
        g = GraphModel()
        node = g.add_node("Array", 100.0, 200.0)
        assert node.x == 100.0
        assert node.y == 200.0

    def test_node_gets_default_properties(self):
        g = GraphModel()
        node = g.add_node("ParameterVector")
        assert "size" in node.properties
        assert node.properties["size"] == 1

    def test_node_gets_auto_name(self):
        g = GraphModel()
        node = g.add_node("Array")
        assert node.properties["name"] == "array_1"

    def test_sequential_auto_names(self):
        g = GraphModel()
        n1 = g.add_node("Array")
        n2 = g.add_node("Array")
        assert n1.properties["name"] == "array_1"
        assert n2.properties["name"] == "array_2"

    def test_auto_names_per_type(self):
        g = GraphModel()
        a = g.add_node("Array")
        p = g.add_node("ParameterVector")
        assert a.properties["name"] == "array_1"
        assert p.properties["name"] == "parameter_vector_1"

    def test_unique_ids(self):
        g = GraphModel()
        ids = {g.add_node("Array").id for _ in range(20)}
        assert len(ids) == 20

    def test_spec_accessible(self):
        g = GraphModel()
        node = g.add_node("Array")
        assert node.spec is NODE_TYPES["Array"]


# ── GraphModel: remove_node ────────────────────────────────────────────


class TestRemoveNode:
    def test_removes_node(self):
        g = GraphModel()
        node = g.add_node("Array")
        g.remove_node(node.id)
        assert node.id not in g.nodes

    def test_removes_connected_edges(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        g.add_edge(arr.id, "data", plf.id, "axis")
        assert len(g.edges) == 1

        g.remove_node(arr.id)
        assert len(g.edges) == 0

    def test_remove_nonexistent_is_noop(self):
        g = GraphModel()
        g.remove_node("nonexistent")  # should not raise


# ── GraphModel: can_connect ────────────────────────────────────────────


class TestCanConnect:
    def test_compatible_types(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        assert g.can_connect(arr.id, "data", plf.id, "axis") is True

    def test_incompatible_types(self):
        g = GraphModel()
        arr = g.add_node("Array")
        fr = g.add_node("FieldRequest")
        # Array output is ARRAY, FieldRequest "field" input expects FIELD
        assert g.can_connect(arr.id, "data", fr.id, "field") is False

    def test_self_connection_rejected(self):
        g = GraphModel()
        arr = g.add_node("Array")
        assert g.can_connect(arr.id, "data", arr.id, "data") is False

    def test_nonexistent_node_rejected(self):
        g = GraphModel()
        arr = g.add_node("Array")
        assert g.can_connect(arr.id, "data", "nonexistent", "axis") is False

    def test_nonexistent_port_rejected(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        assert g.can_connect(arr.id, "wrong_port", plf.id, "axis") is False

    def test_already_connected_input_rejected(self):
        g = GraphModel()
        a1 = g.add_node("Array")
        a2 = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        g.add_edge(a1.id, "data", plf.id, "axis")
        assert g.can_connect(a2.id, "data", plf.id, "axis") is False

    def test_multi_type_port_accepts_compatible(self):
        g = GraphModel()
        arr = g.add_node("Array")
        gl = g.add_node("GaussianLikelihood")
        # sigma accepts ARRAY | UNCERTAINTIES
        assert g.can_connect(arr.id, "data", gl.id, "sigma") is True

    def test_multi_type_port_accepts_other_type(self):
        g = GraphModel()
        cu = g.add_node("ConstantUncertainty")
        gl = g.add_node("GaussianLikelihood")
        # sigma accepts ARRAY | UNCERTAINTIES, ConstantUncertainty outputs UNCERTAINTIES
        assert g.can_connect(cu.id, "uncertainties", gl.id, "sigma") is True

    def test_coordinates_dynamic_ports_connectable(self):
        g = GraphModel()
        arr = g.add_node("Array")
        coords = g.add_node("Coordinates")
        # Default coordinate_names are ["R", "z"]
        assert g.can_connect(arr.id, "data", coords.id, "R") is True
        assert g.can_connect(arr.id, "data", coords.id, "z") is True

    def test_coordinates_nonexistent_dynamic_port_rejected(self):
        g = GraphModel()
        arr = g.add_node("Array")
        coords = g.add_node("Coordinates")
        assert g.can_connect(arr.id, "data", coords.id, "phi") is False


# ── GraphModel: add_edge ──────────────────────────────────────────────


class TestAddEdge:
    def test_adds_edge(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        edge = g.add_edge(arr.id, "data", plf.id, "axis")
        assert edge is not None
        assert edge in g.edges

    def test_returns_none_if_invalid(self):
        g = GraphModel()
        arr = g.add_node("Array")
        edge = g.add_edge(arr.id, "data", arr.id, "data")
        assert edge is None

    def test_edge_fields_correct(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        edge = g.add_edge(arr.id, "data", plf.id, "axis")
        assert edge.source_node_id == arr.id
        assert edge.source_port_name == "data"
        assert edge.target_node_id == plf.id
        assert edge.target_port_name == "axis"


# ── GraphModel: remove_edge ───────────────────────────────────────────


class TestRemoveEdge:
    def test_removes_edge(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        edge = g.add_edge(arr.id, "data", plf.id, "axis")
        g.remove_edge(edge)
        assert edge not in g.edges

    def test_can_reconnect_after_remove(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        edge = g.add_edge(arr.id, "data", plf.id, "axis")
        g.remove_edge(edge)
        edge2 = g.add_edge(arr.id, "data", plf.id, "axis")
        assert edge2 is not None


# ── GraphModel: edges_for_node ─────────────────────────────────────────


class TestEdgesForNode:
    def test_returns_connected_edges(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        edge = g.add_edge(arr.id, "data", plf.id, "axis")
        assert edge in g.edges_for_node(arr.id)
        assert edge in g.edges_for_node(plf.id)

    def test_returns_empty_for_unconnected(self):
        g = GraphModel()
        arr = g.add_node("Array")
        assert g.edges_for_node(arr.id) == []


# ── GraphModel: validate ──────────────────────────────────────────────


class TestValidate:
    def test_no_errors_when_all_connected(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        g.add_edge(arr.id, "data", plf.id, "axis")
        errors = g.validate()
        assert not any("axis" in e for e in errors)

    def test_error_for_unconnected_required_port(self):
        g = GraphModel()
        plf = g.add_node("PiecewiseLinearField")
        errors = g.validate()
        assert any("axis" in e for e in errors)

    def test_no_error_for_optional_port(self):
        g = GraphModel()
        gp = g.add_node("GaussianPrior")
        errors = g.validate()
        # field_request and parameter_vector are optional — shouldn't appear in errors
        assert not any("field_request" in e for e in errors)
        assert not any("parameter_vector" in e for e in errors)

    def test_coordinates_dynamic_ports_validated(self):
        g = GraphModel()
        coords = g.add_node("Coordinates")
        errors = g.validate()
        # R and z are required dynamic ports
        assert any("R" in e for e in errors)
        assert any("z" in e for e in errors)

    def test_coordinates_dynamic_ports_connected_no_error(self):
        g = GraphModel()
        arr_r = g.add_node("Array")
        arr_z = g.add_node("Array")
        coords = g.add_node("Coordinates")
        g.add_edge(arr_r.id, "data", coords.id, "R")
        g.add_edge(arr_z.id, "data", coords.id, "z")
        errors = g.validate()
        coord_errors = [e for e in errors if coords.id in e]
        assert coord_errors == []


# ── NodeModel: effective_input_ports ───────────────────────────────────


class TestEffectiveInputPorts:
    def test_regular_node_returns_spec_ports(self):
        g = GraphModel()
        arr = g.add_node("Array")
        assert arr.effective_input_ports == arr.spec.input_ports

    def test_coordinates_includes_dynamic_ports(self):
        g = GraphModel()
        coords = g.add_node("Coordinates")
        ports = coords.effective_input_ports
        port_names = [p.name for p in ports]
        assert "R" in port_names
        assert "z" in port_names

    def test_coordinates_dynamic_ports_are_array_type(self):
        g = GraphModel()
        coords = g.add_node("Coordinates")
        for port in coords.effective_input_ports:
            assert port.port_types == (PortType.ARRAY,)
            assert port.direction == PortDirection.INPUT

    def test_coordinates_reflects_property_changes(self):
        g = GraphModel()
        coords = g.add_node("Coordinates")
        coords.properties["coordinate_names"] = ["x", "y", "z"]
        port_names = [p.name for p in coords.effective_input_ports]
        assert port_names == ["x", "y", "z"]

    def test_coordinates_empty_names(self):
        g = GraphModel()
        coords = g.add_node("Coordinates")
        coords.properties["coordinate_names"] = []
        assert coords.effective_input_ports == coords.spec.input_ports
