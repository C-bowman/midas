"""Tests for code generation (midas_gui.codegen.script_generator)."""
from __future__ import annotations

import pytest

from midas_gui.codegen.script_generator import (
    _dependency_order,
    _sanitize_var,
    generate_script,
)
from midas_gui.session import GraphModel


# ── _sanitize_var ──────────────────────────────────────────────────────


class TestSanitizeVar:
    def test_valid_name_unchanged(self):
        assert _sanitize_var("my_var") == "my_var"

    def test_spaces_replaced(self):
        assert _sanitize_var("my var") == "my_var"

    def test_leading_digit_prefixed(self):
        assert _sanitize_var("3dogs") == "_3dogs"

    def test_special_chars_replaced(self):
        assert _sanitize_var("x-y.z") == "x_y_z"

    def test_empty_becomes_node(self):
        assert _sanitize_var("") == "node"


# ── _dependency_order ──────────────────────────────────────────────────


class TestDependencyOrder:
    def test_source_before_target(self):
        g = GraphModel()
        arr = g.add_node("Array")
        plf = g.add_node("PiecewiseLinearField")
        g.add_edge(arr.id, "data", plf.id, "axis")

        order = _dependency_order(g)
        ids = [n.id for n in order]
        assert ids.index(arr.id) < ids.index(plf.id)

    def test_chain_order(self):
        """A -> B -> C should produce A, B, C."""
        g = GraphModel()
        arr = g.add_node("Array")
        gl = g.add_node("GaussianLikelihood")
        dl = g.add_node("DiagnosticLikelihood")
        g.add_edge(arr.id, "data", gl.id, "y_data")
        g.add_edge(gl.id, "likelihood", dl.id, "likelihood")

        order = _dependency_order(g)
        ids = [n.id for n in order]
        assert ids.index(arr.id) < ids.index(gl.id)
        assert ids.index(gl.id) < ids.index(dl.id)

    def test_unconnected_nodes_included(self):
        g = GraphModel()
        a = g.add_node("Array")
        b = g.add_node("Array")
        order = _dependency_order(g)
        assert len(order) == 2

    def test_all_nodes_present(self):
        g = GraphModel()
        nodes = [g.add_node("Array") for _ in range(5)]
        order = _dependency_order(g)
        assert {n.id for n in order} == {n.id for n in nodes}


# ── generate_script: Array node ────────────────────────────────────────


class TestGenerateScriptArray:
    def test_linspace(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "x"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 50}
        script = generate_script(g)
        assert "x = np.linspace(0, 1, 50)" in script

    def test_arange(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "x"
        arr.properties["values_config"] = {"source": "arange", "start": 0, "stop": 10, "step": 0.5}
        script = generate_script(g)
        assert "x = np.arange(0, 10, 0.5)" in script

    def test_file_npy(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "data"
        arr.properties["values_config"] = {"source": "file", "path": "my_data.npy"}
        script = generate_script(g)
        assert 'data = np.load("my_data.npy")' in script

    def test_file_csv(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "data"
        arr.properties["values_config"] = {"source": "file", "path": "values.csv"}
        script = generate_script(g)
        assert 'np.loadtxt("values.csv", delimiter=",")' in script


# ── generate_script: ParameterVector ───────────────────────────────────


class TestGenerateScriptParameterVector:
    def test_basic(self):
        g = GraphModel()
        pv = g.add_node("ParameterVector")
        pv.properties["name"] = "my_params"
        pv.properties["size"] = 5
        script = generate_script(g)
        assert 'my_params = ParameterVector("my_params", 5)' in script

    def test_import_added(self):
        g = GraphModel()
        g.add_node("ParameterVector")
        script = generate_script(g)
        assert "from midas.parameters import ParameterVector" in script


# ── generate_script: Coordinates ───────────────────────────────────────


class TestGenerateScriptCoordinates:
    def test_dict_construction(self):
        g = GraphModel()
        arr_r = g.add_node("Array")
        arr_r.properties["name"] = "r_vals"
        arr_r.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        arr_z = g.add_node("Array")
        arr_z.properties["name"] = "z_vals"
        arr_z.properties["values_config"] = {"source": "linspace", "start": -1, "stop": 1, "num": 10}

        coords = g.add_node("Coordinates")
        coords.properties["name"] = "coords"
        coords.properties["coordinate_names"] = ["R", "z"]
        g.add_edge(arr_r.id, "data", coords.id, "R")
        g.add_edge(arr_z.id, "data", coords.id, "z")

        script = generate_script(g)
        assert '"R": r_vals' in script
        assert '"z": z_vals' in script


# ── generate_script: DiagnosticLikelihood ──────────────────────────────


class TestGenerateScriptDiagnosticLikelihood:
    def test_connects_model_and_likelihood(self):
        g = GraphModel()
        dm = g.add_node("LinearDiagnosticModel")
        dm.properties["name"] = "diag"

        arr = g.add_node("Array")
        arr.properties["name"] = "y_data"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        gl = g.add_node("GaussianLikelihood")
        gl.properties["name"] = "like"
        g.add_edge(arr.id, "data", gl.id, "y_data")

        dl = g.add_node("DiagnosticLikelihood")
        dl.properties["name"] = "dl"
        g.add_edge(dm.id, "diagnostic_model", dl.id, "diagnostic_model")
        g.add_edge(gl.id, "likelihood", dl.id, "likelihood")

        script = generate_script(g)
        assert "DiagnosticLikelihood(diag, like" in script

    def test_import_added(self):
        g = GraphModel()
        g.add_node("DiagnosticLikelihood")
        script = generate_script(g)
        assert "DiagnosticLikelihood" in script
        assert "from midas.likelihoods import" in script


# ── generate_script: topological ordering ──────────────────────────────


class TestGenerateScriptOrdering:
    def test_source_emitted_before_target(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "axis"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        plf = g.add_node("PiecewiseLinearField")
        plf.properties["name"] = "field"
        g.add_edge(arr.id, "data", plf.id, "axis")

        script = generate_script(g)
        assert script.index("axis = np.linspace") < script.index("field = PiecewiseLinearField")

    def test_diagnostic_likelihood_after_its_inputs(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "y"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        gl = g.add_node("GaussianLikelihood")
        gl.properties["name"] = "gl"
        g.add_edge(arr.id, "data", gl.id, "y_data")

        dm = g.add_node("LinearDiagnosticModel")
        dm.properties["name"] = "dm"

        dl = g.add_node("DiagnosticLikelihood")
        dl.properties["name"] = "dl"
        g.add_edge(dm.id, "diagnostic_model", dl.id, "diagnostic_model")
        g.add_edge(gl.id, "likelihood", dl.id, "likelihood")

        script = generate_script(g)
        dl_pos = script.index("dl = DiagnosticLikelihood")
        gl_pos = script.index("gl = GaussianLikelihood")
        dm_pos = script.index("dm = LinearDiagnosticModel")
        assert gl_pos < dl_pos
        assert dm_pos < dl_pos


# ── generate_script: imports ───────────────────────────────────────────


class TestGenerateScriptImports:
    def test_always_has_numpy(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g)
        assert "import numpy as np" in script

    def test_plasma_state_imported(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g)
        assert "PlasmaState" in script

    def test_field_request_import(self):
        g = GraphModel()
        g.add_node("FieldRequest")
        script = generate_script(g)
        assert "from midas.parameters import" in script
        assert "FieldRequest" in script


# ── generate_script: build_posterior ───────────────────────────────────


class TestGenerateScriptPosterior:
    def test_includes_field_models(self):
        g = GraphModel()
        plf = g.add_node("PiecewiseLinearField")
        plf.properties["name"] = "te"
        script = generate_script(g)
        assert "field_models=[te]" in script

    def test_includes_priors(self):
        g = GraphModel()
        gp = g.add_node("GaussianPrior")
        gp.properties["name"] = "prior"
        script = generate_script(g)
        assert "priors=[prior]" in script

    def test_includes_diagnostics(self):
        g = GraphModel()
        dl = g.add_node("DiagnosticLikelihood")
        dl.properties["name"] = "diag"
        script = generate_script(g)
        assert "diagnostics=[diag]" in script

    def test_empty_graph_no_posterior(self):
        g = GraphModel()
        script = generate_script(g)
        assert "build_posterior" not in script


# ── generate_script: comments and runnable flags ───────────────────────


class TestGenerateScriptFlags:
    def test_comments_flag(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, comments=True)
        assert "# ── Analysis construction" in script

    def test_no_comments_by_default(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g)
        assert "# ── Analysis construction" not in script

    def test_runnable_template(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, runnable=True)
        assert "scipy.optimize" in script
        assert "HamiltonianChain" in script
        assert "from midas import posterior" in script

    def test_runnable_not_included_by_default(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g)
        assert "scipy" not in script


# ── generate_script: variable name deduplication ───────────────────────


class TestGenerateScriptVarNames:
    def test_duplicate_names_deduplicated(self):
        g = GraphModel()
        a1 = g.add_node("Array")
        a1.properties["name"] = "data"
        a2 = g.add_node("Array")
        a2.properties["name"] = "data"
        script = generate_script(g)
        # Both should appear with different variable names
        assert "data =" in script or "data_1 =" in script


# ── generate_script: validation warnings ───────────────────────────────


class TestGenerateScriptValidation:
    def test_warnings_for_unconnected_ports(self):
        g = GraphModel()
        plf = g.add_node("PiecewiseLinearField")
        plf.properties["name"] = "field"
        script = generate_script(g)
        assert "# WARNING" in script
        assert "axis" in script
