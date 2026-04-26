"""Shared fixtures and test node type registration for gui_tests."""
from midas_gui.session import (
    NODE_TYPES,
    PortType,
    _make_spec,
    _register,
)

# Register minimal node type specs for testing if the auto-discovery
# didn't run (e.g. inference-tools not on PYTHONPATH).

_TEST_SPECS = [
    _make_spec(
        "PiecewiseLinearField", "PiecewiseLinearField", "Field Models",
        [("axis", PortType.ARRAY)],
        [("field", PortType.FIELD)],
        {"name": "", "field_name": "", "axis_name": "psi"},
    ),
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
        "ConstantUncertainty", "ConstantUncertainty", "Uncertainty Models",
        [],
        [("uncertainties", PortType.UNCERTAINTIES)],
        {"name": "", "n_data": 1, "parameter_name": ""},
    ),
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
    _make_spec(
        "LinearDiagnosticModel", "LinearDiagnosticModel", "Diagnostic Models",
        [
            ("field", PortType.FIELD_REQUEST),
            ("model_matrix", PortType.ARRAY),
        ],
        [("diagnostic_model", PortType.DIAGNOSTIC_MODEL)],
        {"name": ""},
    ),
]

for spec in _TEST_SPECS:
    if spec.type_id not in NODE_TYPES:
        _register(spec)
