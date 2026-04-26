"""Tests for session save/load (midas_gui.io.session_io)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from midas_gui.io.session_io import (
    _clean_properties,
    load_session,
    read_imported_modules,
    save_session,
)
from midas_gui.session import GraphModel, NODE_TYPES


# ── _clean_properties ──────────────────────────────────────────────────


class TestCleanProperties:
    def test_preserves_basic_types(self):
        props = {"name": "foo", "size": 3, "ratio": 0.5, "flag": True}
        clean = _clean_properties(props)
        assert clean == props

    def test_preserves_nested_dict(self):
        props = {"values_config": {"source": "linspace", "start": 0, "stop": 1}}
        clean = _clean_properties(props)
        assert clean == props

    def test_preserves_list(self):
        props = {"coordinate_names": ["R", "z"]}
        clean = _clean_properties(props)
        assert clean == props

    def test_strips_numpy_arrays(self):
        props = {"name": "x", "values": np.array([1, 2, 3])}
        clean = _clean_properties(props)
        assert "name" in clean
        assert "values" not in clean

    def test_strips_class_metadata(self):
        props = {"name": "x", "_class": "midas.models.fields.PiecewiseLinearField"}
        clean = _clean_properties(props)
        assert "_class" not in clean

    def test_strips_unresolved_metadata(self):
        props = {"name": "x", "_unresolved": {"covariance": {"type_name": "Cov"}}}
        clean = _clean_properties(props)
        assert "_unresolved" not in clean

    def test_preserves_none_values(self):
        props = {"name": "x", "values": None}
        clean = _clean_properties(props)
        assert clean["values"] is None

    def test_preserves_tuple(self):
        props = {"limits": (0.0, 1.0)}
        clean = _clean_properties(props)
        assert clean["limits"] == (0.0, 1.0)


# ── save_session ───────────────────────────────────────────────────────


class TestSaveSession:
    def test_creates_file(self, tmp_path):
        g = GraphModel()
        g.add_node("Array")
        path = tmp_path / "test.midas"
        save_session(g, path)
        assert path.exists()

    def test_valid_json(self, tmp_path):
        g = GraphModel()
        g.add_node("Array")
        path = tmp_path / "test.midas"
        save_session(g, path)
        data = json.loads(path.read_text())
        assert "version" in data
        assert "nodes" in data
        assert "edges" in data

    def test_saves_node_data(self, tmp_path):
        g = GraphModel()
        node = g.add_node("Array", 100.0, 200.0)
        node.properties["name"] = "my_array"
        path = tmp_path / "test.midas"
        save_session(g, path)

        data = json.loads(path.read_text())
        assert len(data["nodes"]) == 1
        saved = data["nodes"][0]
        assert saved["id"] == node.id
        assert saved["type_id"] == "Array"
        assert saved["x"] == 100.0
        assert saved["y"] == 200.0
        assert saved["properties"]["name"] == "my_array"

    def test_saves_edges(self, tmp_path):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        g.add_edge(arr.id, "data", plf.id, "axis")
        path = tmp_path / "test.midas"
        save_session(g, path)

        data = json.loads(path.read_text())
        assert len(data["edges"]) == 1
        edge = data["edges"][0]
        assert edge["source_node_id"] == arr.id
        assert edge["source_port_name"] == "data"
        assert edge["target_node_id"] == plf.id
        assert edge["target_port_name"] == "axis"

    def test_saves_name_counters(self, tmp_path):
        g = GraphModel()
        g.add_node("Array")
        g.add_node("Array")
        path = tmp_path / "test.midas"
        save_session(g, path)

        data = json.loads(path.read_text())
        assert data["name_counters"]["Array"] == 2

    def test_saves_imported_modules(self, tmp_path):
        g = GraphModel()
        path = tmp_path / "test.midas"
        save_session(g, path, ["/some/module.py", "/other/mod.py"])

        data = json.loads(path.read_text())
        assert data["imported_modules"] == ["/some/module.py", "/other/mod.py"]

    def test_no_imported_modules_saves_empty_list(self, tmp_path):
        g = GraphModel()
        path = tmp_path / "test.midas"
        save_session(g, path)

        data = json.loads(path.read_text())
        assert data["imported_modules"] == []

    def test_numpy_arrays_excluded_from_json(self, tmp_path):
        g = GraphModel()
        node = g.add_node("Array")
        node.properties["my_array"] = np.linspace(0, 1, 100)
        path = tmp_path / "test.midas"
        save_session(g, path)

        # Should not raise — numpy arrays were stripped
        data = json.loads(path.read_text())
        saved_props = data["nodes"][0]["properties"]
        assert "my_array" not in saved_props


# ── read_imported_modules ──────────────────────────────────────────────


class TestReadImportedModules:
    def test_reads_modules(self, tmp_path):
        g = GraphModel()
        path = tmp_path / "test.midas"
        save_session(g, path, ["/a.py", "/b.py"])
        assert read_imported_modules(path) == ["/a.py", "/b.py"]

    def test_returns_empty_if_missing(self, tmp_path):
        path = tmp_path / "test.midas"
        path.write_text(json.dumps({"version": "1.0", "nodes": [], "edges": []}))
        assert read_imported_modules(path) == []


# ── load_session ───────────────────────────────────────────────────────


class TestLoadSession:
    def test_round_trip_nodes(self, tmp_path):
        g = GraphModel()
        arr = g.add_node("Array", 50.0, 100.0)
        arr.properties["name"] = "psi_axis"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 50}

        path = tmp_path / "test.midas"
        save_session(g, path)
        g2, _ = load_session(path)

        assert len(g2.nodes) == 1
        loaded = g2.nodes[arr.id]
        assert loaded.type_id == "Array"
        assert loaded.x == 50.0
        assert loaded.y == 100.0
        assert loaded.properties["name"] == "psi_axis"
        assert loaded.properties["values_config"]["source"] == "linspace"

    def test_round_trip_edges(self, tmp_path):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        g.add_edge(arr.id, "data", plf.id, "axis")

        path = tmp_path / "test.midas"
        save_session(g, path)
        g2, _ = load_session(path)

        assert len(g2.edges) == 1
        edge = g2.edges[0]
        assert edge.source_node_id == arr.id
        assert edge.target_node_id == plf.id

    def test_round_trip_name_counters(self, tmp_path):
        g = GraphModel()
        g.add_node("Array")
        g.add_node("Array")
        g.add_node("ParameterVector")

        path = tmp_path / "test.midas"
        save_session(g, path)
        g2, _ = load_session(path)

        assert g2._name_counters["Array"] == 2
        assert g2._name_counters["ParameterVector"] == 1

    def test_round_trip_imported_modules(self, tmp_path):
        g = GraphModel()
        path = tmp_path / "test.midas"
        save_session(g, path, ["/my/module.py"])
        _, modules = load_session(path)
        assert modules == ["/my/module.py"]

    def test_skips_unknown_node_types(self, tmp_path):
        path = tmp_path / "test.midas"
        data = {
            "version": "1.0",
            "name_counters": {},
            "imported_modules": [],
            "nodes": [
                {"id": "abc123", "type_id": "NonExistentType", "x": 0, "y": 0, "properties": {}},
                {"id": "def456", "type_id": "Array", "x": 0, "y": 0, "properties": {"name": "ok"}},
            ],
            "edges": [],
        }
        path.write_text(json.dumps(data))
        g, _ = load_session(path)

        assert len(g.nodes) == 1
        assert "def456" in g.nodes
        assert "abc123" not in g.nodes

    def test_skips_edges_with_missing_nodes(self, tmp_path):
        path = tmp_path / "test.midas"
        data = {
            "version": "1.0",
            "name_counters": {},
            "imported_modules": [],
            "nodes": [
                {"id": "aaa", "type_id": "Array", "x": 0, "y": 0, "properties": {"name": "x"}},
            ],
            "edges": [
                {
                    "source_node_id": "aaa",
                    "source_port_name": "data",
                    "target_node_id": "missing",
                    "target_port_name": "axis",
                },
            ],
        }
        path.write_text(json.dumps(data))
        g, _ = load_session(path)

        assert len(g.edges) == 0

    def test_restores_spec_defaults_for_missing_props(self, tmp_path):
        """Properties not saved should be restored from the spec defaults."""
        path = tmp_path / "test.midas"
        data = {
            "version": "1.0",
            "name_counters": {},
            "imported_modules": [],
            "nodes": [
                {"id": "abc", "type_id": "ParameterVector", "x": 0, "y": 0,
                 "properties": {"name": "pv1"}},
            ],
            "edges": [],
        }
        path.write_text(json.dumps(data))
        g, _ = load_session(path)

        node = g.nodes["abc"]
        # "size" wasn't in saved properties but should be restored from defaults
        assert node.properties["size"] == 1

    def test_saved_props_override_defaults(self, tmp_path):
        path = tmp_path / "test.midas"
        data = {
            "version": "1.0",
            "name_counters": {},
            "imported_modules": [],
            "nodes": [
                {"id": "abc", "type_id": "ParameterVector", "x": 0, "y": 0,
                 "properties": {"name": "pv1", "size": 42}},
            ],
            "edges": [],
        }
        path.write_text(json.dumps(data))
        g, _ = load_session(path)

        assert g.nodes["abc"].properties["size"] == 42

    def test_coordinates_dynamic_ports_survive(self, tmp_path):
        g = GraphModel()
        coords = g.add_node("Coordinates")
        coords.properties["coordinate_names"] = ["x", "y", "z"]

        path = tmp_path / "test.midas"
        save_session(g, path)
        g2, _ = load_session(path)

        loaded = list(g2.nodes.values())[0]
        assert loaded.properties["coordinate_names"] == ["x", "y", "z"]
        port_names = [p.name for p in loaded.effective_input_ports]
        assert port_names == ["x", "y", "z"]

    def test_empty_graph_round_trip(self, tmp_path):
        g = GraphModel()
        path = tmp_path / "test.midas"
        save_session(g, path)
        g2, _ = load_session(path)
        assert len(g2.nodes) == 0
        assert len(g2.edges) == 0
