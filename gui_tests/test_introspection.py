"""Tests for the introspection engine (midas_gui.introspection)."""
from __future__ import annotations

import pytest
from numpy import ndarray

from midas_gui.session import PortDirection, PortType

# These tests require the full MIDAS + inference-tools import chain
try:
    from midas_gui.introspection import (
        _classify_annotation,
        _is_coordinates_annotation,
        discover_builtin_nodes,
        discover_user_module,
        generate_node_spec,
    )
    _HAS_INTROSPECTION = True
except Exception:
    _HAS_INTROSPECTION = False

pytestmark = pytest.mark.skipif(
    not _HAS_INTROSPECTION,
    reason="Requires MIDAS and inference-tools to be importable",
)


# ── Annotation classification ──────────────────────────────────────────


class TestClassifyAnnotation:
    def test_ndarray_becomes_array_port(self):
        role, detail = _classify_annotation(ndarray)
        assert role == "port"
        assert detail == (PortType.ARRAY,)

    def test_str_becomes_config(self):
        role, detail = _classify_annotation(str)
        assert role == "config"
        assert detail is str

    def test_int_becomes_config(self):
        role, detail = _classify_annotation(int)
        assert role == "config"
        assert detail is int

    def test_float_becomes_config(self):
        role, detail = _classify_annotation(float)
        assert role == "config"
        assert detail is float

    def test_tuple_becomes_config(self):
        role, detail = _classify_annotation(tuple[float, float])
        assert role == "config"
        assert detail is tuple

    def test_parameter_vector_becomes_params_port(self):
        from midas.parameters import ParameterVector
        role, detail = _classify_annotation(ParameterVector)
        assert role == "port"
        assert detail == (PortType.PARAMS,)

    def test_field_request_becomes_field_request_port(self):
        from midas.parameters import FieldRequest
        role, detail = _classify_annotation(FieldRequest)
        assert role == "port"
        assert detail == (PortType.FIELD_REQUEST,)

    def test_uncertainty_model_becomes_uncertainties_port(self):
        from midas.likelihoods.uncertainties import UncertaintyModel
        role, detail = _classify_annotation(UncertaintyModel)
        assert role == "port"
        assert detail == (PortType.UNCERTAINTIES,)

    def test_coordinates_becomes_coordinates_port(self):
        from midas.parameters import Coordinates
        role, detail = _classify_annotation(Coordinates)
        assert role == "port"
        assert detail == (PortType.COORDINATES,)

    def test_union_ndarray_uncertainty_becomes_multi_type_port(self):
        from midas.likelihoods.uncertainties import UncertaintyModel
        ann = ndarray | UncertaintyModel
        role, detail = _classify_annotation(ann)
        assert role == "port"
        assert set(detail) == {PortType.ARRAY, PortType.UNCERTAINTIES}

    def test_unknown_type_becomes_unresolved(self):
        class SomeCustomType:
            pass
        role, detail = _classify_annotation(SomeCustomType)
        assert role == "unresolved"
        assert detail is None


# ── Coordinates annotation detection ───────────────────────────────────


class TestIsCoordinatesAnnotation:
    def test_coordinates_alias_matches(self):
        from midas.parameters import Coordinates
        assert _is_coordinates_annotation(Coordinates) is True

    def test_plain_dict_str_ndarray_matches(self):
        assert _is_coordinates_annotation(dict[str, ndarray]) is True

    def test_other_dict_does_not_match(self):
        assert _is_coordinates_annotation(dict[str, str]) is False

    def test_non_dict_does_not_match(self):
        assert _is_coordinates_annotation(str) is False


# ── generate_node_spec ─────────────────────────────────────────────────


class TestGenerateNodeSpec:
    def _abc_info_for(self, abc_name: str):
        """Resolve the ABC info tuple for a given ABC name."""
        from midas_gui.introspection import _resolve_abc_classes
        for abc_cls, out_name, out_type, category in _resolve_abc_classes():
            if abc_cls.__name__ == abc_name:
                return (abc_cls, out_name, out_type, category)
        raise ValueError(f"Unknown ABC: {abc_name}")

    def test_skips_abstract_classes(self):
        from midas.models.fields import FieldModel
        info = self._abc_info_for("FieldModel")
        result = generate_node_spec(FieldModel, info)
        assert result is None

    def test_piecewise_linear_field(self):
        from midas.models.fields import PiecewiseLinearField
        info = self._abc_info_for("FieldModel")
        spec = generate_node_spec(PiecewiseLinearField, info)

        assert spec is not None
        assert spec.type_id == "PiecewiseLinearField"
        assert spec.category == "Field Models"

        # Should have one input port: axis (ARRAY)
        assert len(spec.input_ports) == 1
        axis_port = spec.input_ports[0]
        assert axis_port.name == "axis"
        assert axis_port.port_types == (PortType.ARRAY,)
        assert axis_port.direction == PortDirection.INPUT
        assert axis_port.required is True

        # Should have one output port: field (FIELD)
        assert len(spec.output_ports) == 1
        assert spec.output_ports[0].name == "field"
        assert spec.output_ports[0].port_types == (PortType.FIELD,)

        # Config properties: field_name (str) and axis_name (str)
        assert "field_name" in spec.default_properties
        assert "axis_name" in spec.default_properties

    def test_gaussian_likelihood(self):
        from midas.likelihoods import GaussianLikelihood
        info = self._abc_info_for("LikelihoodFunction")
        spec = generate_node_spec(GaussianLikelihood, info)

        assert spec is not None
        assert spec.type_id == "GaussianLikelihood"

        # Two input ports: y_data (ARRAY), sigma (ARRAY | UNCERTAINTIES)
        assert len(spec.input_ports) == 2
        y_port = spec.input_ports[0]
        assert y_port.name == "y_data"
        assert y_port.port_types == (PortType.ARRAY,)

        sigma_port = spec.input_ports[1]
        assert sigma_port.name == "sigma"
        assert set(sigma_port.port_types) == {PortType.ARRAY, PortType.UNCERTAINTIES}

        # Output: likelihood (LIKELIHOOD)
        assert len(spec.output_ports) == 1
        assert spec.output_ports[0].port_types == (PortType.LIKELIHOOD,)

    def test_gaussian_prior_has_optional_ports(self):
        from midas.priors import GaussianPrior
        info = self._abc_info_for("BasePrior")
        spec = generate_node_spec(GaussianPrior, info)

        assert spec is not None
        # Priors are endpoints — no output ports
        assert len(spec.output_ports) == 0

        port_map = {p.name: p for p in spec.input_ports}

        # Required ports
        assert port_map["mean"].required is True
        assert port_map["standard_deviation"].required is True

        # Optional ports (default=None)
        assert port_map["field_request"].required is False
        assert port_map["parameter_vector"].required is False

    def test_constant_uncertainty_has_config_props(self):
        from midas.likelihoods.uncertainties import ConstantUncertainty
        info = self._abc_info_for("UncertaintyModel")
        spec = generate_node_spec(ConstantUncertainty, info)

        assert spec is not None
        # n_data (int) and parameter_name (str) should be config properties
        assert "n_data" in spec.default_properties
        assert "parameter_name" in spec.default_properties
        # No input ports (all params are config)
        assert len(spec.input_ports) == 0

    def test_triangular_mesh_field_has_coordinates_port(self):
        from midas.models.fields import TriangularMeshField
        info = self._abc_info_for("FieldModel")
        spec = generate_node_spec(TriangularMeshField, info)

        assert spec is not None
        port_map = {p.name: p for p in spec.input_ports}
        assert "mesh_coordinates" in port_map
        assert port_map["mesh_coordinates"].port_types == (PortType.COORDINATES,)

    def test_class_metadata_stored(self):
        from midas.models.fields import PiecewiseLinearField
        info = self._abc_info_for("FieldModel")
        spec = generate_node_spec(PiecewiseLinearField, info)

        assert "_class" in spec.default_properties
        assert spec.default_properties["_class"] == "midas.models.fields.PiecewiseLinearField"

    def test_beta_prior_tuple_config(self):
        from midas.priors import BetaPrior
        info = self._abc_info_for("BasePrior")
        spec = generate_node_spec(BetaPrior, info)

        assert spec is not None
        assert "limits" in spec.default_properties
        assert spec.default_properties["limits"] == (0, 1)


# ── discover_builtin_nodes ─────────────────────────────────────────────


class TestDiscoverBuiltinNodes:
    def test_discovers_expected_classes(self):
        specs = discover_builtin_nodes()

        expected = {
            "PiecewiseLinearField", "CubicSplineField", "BSplineField", "ExSplineField",
            "TriangularMeshField",
            "LinearDiagnosticModel",
            "GaussianLikelihood", "LogisticLikelihood", "CauchyLikelihood",
            "ConstantUncertainty", "LinearUncertainty",
            "GaussianPrior", "ExponentialPrior", "BetaPrior",
            "GaussianProcessPrior", "SoftLimitPrior",
        }
        assert expected.issubset(set(specs.keys()))

    def test_does_not_include_abcs(self):
        specs = discover_builtin_nodes()
        abc_names = {"FieldModel", "DiagnosticModel", "LikelihoodFunction",
                     "UncertaintyModel", "BasePrior"}
        assert abc_names.isdisjoint(set(specs.keys()))

    def test_categories_correct(self):
        specs = discover_builtin_nodes()
        assert specs["PiecewiseLinearField"].category == "Field Models"
        assert specs["LinearDiagnosticModel"].category == "Diagnostic Models"
        assert specs["GaussianLikelihood"].category == "Likelihoods"
        assert specs["ConstantUncertainty"].category == "Uncertainty Models"
        assert specs["GaussianPrior"].category == "Priors"


# ── discover_user_module ───────────────────────────────────────────────


class TestDiscoverUserModule:
    def test_discovers_test_utilities(self):
        import os
        util_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "tests", "utilities.py"
        )
        specs = discover_user_module(os.path.abspath(util_path))

        assert "StraightLine" in specs
        assert "Polynomial" in specs

    def test_straight_line_spec(self):
        import os
        util_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "tests", "utilities.py"
        )
        specs = discover_user_module(os.path.abspath(util_path))
        spec = specs["StraightLine"]

        assert spec.category == "Diagnostic Models"
        assert len(spec.input_ports) == 1
        assert spec.input_ports[0].name == "x_axis"
        assert spec.input_ports[0].port_types == (PortType.ARRAY,)
        assert len(spec.output_ports) == 1
        assert spec.output_ports[0].port_types == (PortType.DIAGNOSTIC_MODEL,)

    def test_polynomial_has_order_config(self):
        import os
        util_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "tests", "utilities.py"
        )
        specs = discover_user_module(os.path.abspath(util_path))
        spec = specs["Polynomial"]

        assert "order" in spec.default_properties
        assert spec.default_properties["order"] == 2

    def test_nonexistent_file_returns_empty(self):
        specs = discover_user_module("/nonexistent/path/fake.py")
        assert specs == {}

    def test_file_with_no_abc_subclasses_returns_empty(self, tmp_path):
        f = tmp_path / "empty_module.py"
        f.write_text("x = 42\n")
        specs = discover_user_module(str(f))
        assert specs == {}
