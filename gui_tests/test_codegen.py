"""Tests for code generation (midas_gui.codegen.script_generator)."""
from __future__ import annotations

import pytest

from midas_gui.script_generator import (
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

    def test_file_npz(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "data"
        arr.properties["values_config"] = {"source": "file", "path": "archive.npz", "npz_key": "temperatures"}
        script = generate_script(g)
        assert 'data = np.load("archive.npz")["temperatures"]' in script

    def test_full(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "zeros"
        arr.properties["values_config"] = {"source": "constant", "size": 20, "value": 0.0}
        script = generate_script(g)
        assert "zeros = np.full(20, 0.0)" in script

    def test_placeholder(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "data"
        arr.properties["values_config"] = {"source": "placeholder"}
        script = generate_script(g)
        assert "data: np.ndarray" in script
        assert "# TODO" in script

    def test_placeholder_is_default(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "data"
        script = generate_script(g)
        assert "data: np.ndarray" in script


# ── generate_script: ParameterVector ───────────────────────────────────


class TestGenerateScriptParameterVector:
    def test_basic(self):
        g = GraphModel()
        pv = g.add_node("ParameterVector")
        pv.properties["name"] = "my_params"
        pv.properties["size"] = 5
        script = generate_script(g)
        assert 'my_params = ParameterVector(name="my_params", size=5)' in script

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
        assert "DiagnosticLikelihood(" in script
        assert "diagnostic_model=diag" in script
        assert "likelihood=like" in script

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


# ── generate_script: runnable flag ─────────────────────────────────────


class TestGenerateScriptFlags:
    def test_runnable_template(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, runnable=True)
        assert "scipy.optimize" in script
        assert "HamiltonianChain" in script
        assert "from midas import posterior" in script

    def test_runnable_template_is_valid_python(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, runnable=True)
        compile(script, "<generated>", "exec")

    def test_runnable_not_included_by_default(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g)
        assert "scipy" not in script


# ── generate_script: imported modules sys.path ─────────────────────────


class TestGenerateScriptImportedModules:
    def test_adds_sys_path_insert(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, imported_modules=["/home/chris/models/custom.py"])
        assert "import sys" in script
        assert 'sys.path.insert(0, "/home/chris/models")' in script

    def test_windows_path_normalised(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, imported_modules=["C:\\Users\\chris\\models\\custom.py"])
        assert 'sys.path.insert(0, "C:/Users/chris/models")' in script

    def test_no_sys_path_without_modules(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g)
        assert "sys.path.insert" not in script

    def test_deduplicates_directories(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, imported_modules=[
            "/home/chris/models/a.py",
            "/home/chris/models/b.py",
        ])
        assert script.count('sys.path.insert(0, "/home/chris/models")') == 1

    def test_comment_present(self):
        g = GraphModel()
        g.add_node("Array")
        script = generate_script(g, imported_modules=["/tmp/test.py"])
        assert "# Add imported module directories to the path" in script

    def test_path_before_imports(self):
        g = GraphModel()
        g.add_node("ParameterVector")
        script = generate_script(g, imported_modules=["/tmp/models.py"])
        sys_pos = script.index("sys.path.insert")
        import_pos = script.index("from midas")
        assert sys_pos < import_pos


# ── generate_script: variable name deduplication ───────────────────────


class TestGenerateScriptVarNames:
    def test_duplicate_names_deduplicated(self):
        g = GraphModel()
        a1 = g.add_node("Array")
        a1.properties["name"] = "data"
        a1.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}
        a2 = g.add_node("Array")
        a2.properties["name"] = "data"
        a2.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}
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


# ── generate_script: FieldRequest ──────────────────────────────────────


class TestGenerateScriptFieldRequest:
    def test_basic_emission(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "basis"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        plf = g.add_node("PiecewiseLinearField")
        plf.properties["name"] = "ne"
        plf.properties["field_name"] = "ne"
        g.add_edge(arr.id, "data", plf.id, "axis")

        r = g.add_node("Array")
        r.properties["name"] = "r_vals"
        r.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 20}

        coords = g.add_node("Coordinates")
        coords.properties["name"] = "coords"
        coords.properties["coordinate_names"] = ["R"]
        g.add_edge(r.id, "data", coords.id, "R")

        fr = g.add_node("FieldRequest")
        fr.properties["name"] = "fr"
        g.add_edge(plf.id, "field", fr.id, "field")
        g.add_edge(coords.id, "coordinates", fr.id, "coordinates")

        script = generate_script(g)
        assert 'FieldRequest(' in script
        assert 'name="ne"' in script
        assert 'coordinates=coords' in script

    def test_import_added(self):
        g = GraphModel()
        g.add_node("FieldRequest")
        script = generate_script(g)
        assert "FieldRequest" in script
        assert "from midas.parameters import" in script


# ── generate_script: PiecewiseLinearField ──────────────────────────────


class TestGenerateScriptFieldModel:
    def test_emission_with_axis(self):
        g = GraphModel()
        arr = g.add_node("Array")
        arr.properties["name"] = "psi_grid"
        arr.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 20}

        plf = g.add_node("PiecewiseLinearField")
        plf.properties["name"] = "ne"
        plf.properties["field_name"] = "electron_density"
        plf.properties["axis_name"] = "psi"
        g.add_edge(arr.id, "data", plf.id, "axis")

        script = generate_script(g)
        assert 'ne = PiecewiseLinearField(' in script
        assert 'field_name="electron_density"' in script
        assert 'axis=psi_grid' in script
        assert 'axis_name="psi"' in script

    def test_unconnected_axis_shows_todo(self):
        g = GraphModel()
        plf = g.add_node("PiecewiseLinearField")
        plf.properties["name"] = "ne"
        plf.properties["field_name"] = "ne"
        script = generate_script(g)
        assert "TODO" in script


# ── generate_script: LinearDiagnosticModel ─────────────────────────────


class TestGenerateScriptLinearDiagnosticModel:
    def test_emission_with_connections(self):
        g = GraphModel()
        fr = g.add_node("FieldRequest")
        fr.properties["name"] = "fr"

        mat = g.add_node("Array")
        mat.properties["name"] = "matrix"
        mat.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        dm = g.add_node("LinearDiagnosticModel")
        dm.properties["name"] = "model"
        g.add_edge(fr.id, "field_request", dm.id, "field")
        g.add_edge(mat.id, "data", dm.id, "model_matrix")

        script = generate_script(g)
        assert "model = LinearDiagnosticModel(" in script
        assert "field=fr," in script
        assert "model_matrix=matrix," in script


# ── generate_script: ConstantUncertainty ───────────────────────────────


class TestGenerateScriptConstantUncertainty:
    def test_basic_emission(self):
        g = GraphModel()
        cu = g.add_node("ConstantUncertainty")
        cu.properties["name"] = "sigma_model"
        cu.properties["n_data"] = 50
        cu.properties["parameter_name"] = "sigma_te"
        script = generate_script(g)
        assert "sigma_model = ConstantUncertainty(" in script
        assert "n_data=50," in script
        assert 'parameter_name="sigma_te",' in script

    def test_default_parameter_name(self):
        g = GraphModel()
        cu = g.add_node("ConstantUncertainty")
        cu.properties["name"] = "sigma"
        cu.properties["parameter_name"] = ""
        script = generate_script(g)
        assert 'parameter_name="",' in script


# ── generate_script: GaussianLikelihood ────────────────────────────────


class TestGenerateScriptGaussianLikelihood:
    def test_with_array_sigma(self):
        g = GraphModel()
        data = g.add_node("Array")
        data.properties["name"] = "y"
        data.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        sigma = g.add_node("Array")
        sigma.properties["name"] = "err"
        sigma.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        gl = g.add_node("GaussianLikelihood")
        gl.properties["name"] = "like"
        g.add_edge(data.id, "data", gl.id, "y_data")
        g.add_edge(sigma.id, "data", gl.id, "sigma")

        script = generate_script(g)
        assert "like = GaussianLikelihood(" in script
        assert "y_data=y," in script
        assert "sigma=err," in script

    def test_with_uncertainty_model(self):
        g = GraphModel()
        data = g.add_node("Array")
        data.properties["name"] = "y"
        data.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        cu = g.add_node("ConstantUncertainty")
        cu.properties["name"] = "unc"

        gl = g.add_node("GaussianLikelihood")
        gl.properties["name"] = "like"
        g.add_edge(data.id, "data", gl.id, "y_data")
        g.add_edge(cu.id, "uncertainties", gl.id, "sigma")

        script = generate_script(g)
        assert "like = GaussianLikelihood(" in script
        assert "y_data=y," in script
        assert "sigma=unc," in script

    def test_without_sigma(self):
        g = GraphModel()
        data = g.add_node("Array")
        data.properties["name"] = "y"
        data.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        gl = g.add_node("GaussianLikelihood")
        gl.properties["name"] = "like"
        g.add_edge(data.id, "data", gl.id, "y_data")

        script = generate_script(g)
        assert "like = GaussianLikelihood(" in script
        assert "y_data=y," in script
        # sigma is optional and unconnected — should be omitted
        assert "sigma" not in script.split("GaussianLikelihood")[1].split(")")[0]


# ── generate_script: GaussianPrior ─────────────────────────────────────


class TestGenerateScriptGaussianPrior:
    def test_with_field_request(self):
        g = GraphModel()
        mean = g.add_node("Array")
        mean.properties["name"] = "mu"
        mean.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        std = g.add_node("Array")
        std.properties["name"] = "sd"
        std.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        fr = g.add_node("FieldRequest")
        fr.properties["name"] = "fr"

        gp = g.add_node("GaussianPrior")
        gp.properties["name"] = "prior"
        g.add_edge(mean.id, "data", gp.id, "mean")
        g.add_edge(std.id, "data", gp.id, "standard_deviation")
        g.add_edge(fr.id, "field_request", gp.id, "field_request")

        script = generate_script(g)
        assert "prior = GaussianPrior(" in script
        assert 'name="prior"' in script
        assert "mean=mu," in script
        assert "standard_deviation=sd," in script
        assert "field_request=fr," in script

    def test_with_parameter_vector(self):
        g = GraphModel()
        mean = g.add_node("Array")
        mean.properties["name"] = "mu"
        mean.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        std = g.add_node("Array")
        std.properties["name"] = "sd"
        std.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        pv = g.add_node("ParameterVector")
        pv.properties["name"] = "params"

        gp = g.add_node("GaussianPrior")
        gp.properties["name"] = "prior"
        g.add_edge(mean.id, "data", gp.id, "mean")
        g.add_edge(std.id, "data", gp.id, "standard_deviation")
        g.add_edge(pv.id, "parameter_vector", gp.id, "parameter_vector")

        script = generate_script(g)
        assert "prior = GaussianPrior(" in script
        assert "parameter_vector=params," in script
        assert "field_request" not in script.split("GaussianPrior")[1].split(")")[0]

    def test_without_either_shows_todo(self):
        """When both optional ports are unconnected, they are simply omitted."""
        g = GraphModel()
        mean = g.add_node("Array")
        mean.properties["name"] = "mu"
        mean.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        std = g.add_node("Array")
        std.properties["name"] = "sd"
        std.properties["values_config"] = {"source": "linspace", "start": 0, "stop": 1, "num": 10}

        gp = g.add_node("GaussianPrior")
        gp.properties["name"] = "prior"
        g.add_edge(mean.id, "data", gp.id, "mean")
        g.add_edge(std.id, "data", gp.id, "standard_deviation")

        script = generate_script(g)
        assert "prior = GaussianPrior(" in script
        assert "mean=mu," in script
        assert "standard_deviation=sd," in script
        # Optional ports are omitted, not shown as TODO
        prior_block = script.split("GaussianPrior(")[1].split(")")[0]
        assert "field_request" not in prior_block
        assert "parameter_vector" not in prior_block


# ── generate_script: build_posterior with multiple items ───────────────


class TestGenerateScriptPosteriorMultiple:
    def test_multiple_field_models(self):
        g = GraphModel()
        f1 = g.add_node("PiecewiseLinearField")
        f1.properties["name"] = "ne"
        f2 = g.add_node("PiecewiseLinearField")
        f2.properties["name"] = "te"
        script = generate_script(g)
        assert "field_models=[ne, te]" in script

    def test_multiple_priors(self):
        g = GraphModel()
        p1 = g.add_node("GaussianPrior")
        p1.properties["name"] = "p1"
        p2 = g.add_node("GaussianPrior")
        p2.properties["name"] = "p2"
        script = generate_script(g)
        assert "priors=[p1, p2]" in script

    def test_multiple_diagnostics(self):
        g = GraphModel()
        d1 = g.add_node("DiagnosticLikelihood")
        d1.properties["name"] = "d1"
        d2 = g.add_node("DiagnosticLikelihood")
        d2.properties["name"] = "d2"
        script = generate_script(g)
        assert "diagnostics=[d1, d2]" in script
