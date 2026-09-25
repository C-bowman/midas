from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from utilities import StraightLine
from midas import FieldRequest
from midas.validation.gradients import (
    GradientReport,
    _colorize,
    estimate_gradient_errors,
    finite_difference_gradient,
    validate_gradient,
    vector_error_metrics,
)
from midas.likelihoods import GaussianLikelihood
from midas.models.fields import PiecewiseLinearField
from midas.priors import GaussianPrior
from midas.state import DiagnosticLikelihood, PlasmaState


@pytest.mark.parametrize("n_passes, status, rate", [
    (5, "PASS", "100.0%"), (3, "FAIL", "60.0%"), (0, "FAIL", "0.0%"),
])
def test_sampled_report_representation(n_passes, status, rate):
    report = GradientReport(
        success=n_passes == 5,
        results={"diagnostic": {"density": {
            "max_dir_err": 0.00123, "max_mag_err": 0.00456, "n_passes": n_passes,
        }}},
        n_samples=5,
    )
    assert repr(report) == (
        f"Gradient validation: {status}\n\n"
        "[diagnostic]\n"
        "\tParameter | Status | Max direction error | Max magnitude error | Pass rate\n"
        "\t----------+--------+---------------------+---------------------+----------\n"
        f"\tdensity   | {status:<6} |             0.00123 |             0.00456 | {rate:>9}"
    )


def test_sampled_report_empty_sample():
    report = GradientReport(False, {"prior": {"density": {
        "max_dir_err": float("inf"), "max_mag_err": float("inf"), "n_passes": 0,
    }}}, 0)
    assert "density   | FAIL" in repr(report)
    assert "|                 inf |                 inf |       N/A" in repr(report)


def test_sampled_report_empty_component():
    report = GradientReport(False, {"prior": {}}, 0)
    assert repr(report) == (
        "Gradient validation: FAIL\n\n"
        "[prior]\n"
        "\tParameter | Status | Max direction error | Max magnitude error | Pass rate\n"
        "\t----------+--------+---------------------+---------------------+----------"
    )


@pytest.mark.parametrize("success", [True, False])
def test_sampled_report_truthiness_reflects_success(success):
    report = GradientReport(success, {}, 0)
    assert bool(report) is success


def test_sampled_report_color_is_optional():
    report = GradientReport(True, {"prior": {
        "passing": {"max_dir_err": 0.0, "max_mag_err": 0.0, "n_passes": 1},
        "failing": {"max_dir_err": 1.0, "max_mag_err": 1.0, "n_passes": 0},
    }}, 1)

    colored = report.format(color=True)

    assert "Gradient validation: \033[32mPASS\033[0m" in colored
    assert "\n\n\033[36m[prior]\033[0m\n" in colored
    assert "\033[32mPASS  \033[0m" in colored
    assert "\033[31mFAIL  \033[0m" in colored
    assert "\033[" not in repr(report)


@pytest.mark.parametrize("color", [True, False])
def test_print_report_defaults_to_color(capsys, color):
    report = GradientReport(True, {"prior": {}}, 1)

    if color:
        report.print_report()
    else:
        report.print_report(color=False)

    output = capsys.readouterr().out
    assert output == report.format(color=color) + "\n"
    assert ("\033[" in output) is color


def test_colorize_accepts_color_name_or_ansi_code():
    assert _colorize("warning", "YELLOW") == "\033[33mwarning\033[0m"
    assert _colorize("custom", 38) == "\033[38mcustom\033[0m"


def test_colorize_rejects_unknown_color_name():
    with pytest.raises(ValueError, match="Unknown ANSI color 'orange'"):
        _colorize("warning", "orange")


@pytest.mark.parametrize("param_slice", [slice(0, 3), slice(0, 2), slice(1, 3), slice(2, 3)])
def test_finite_difference_gradient(param_slice):
    point = np.array([0.0, -2.0, 1000.0])
    original = point.copy()
    weights = np.array([1.0, 2.0, 3.0])
    evaluations = []

    def quadratic(values):
        evaluations.append(values.copy())
        return np.sum(weights * values**2)

    numerical = finite_difference_gradient(quadratic, point, param_slice, 1e-4)

    assert_allclose(numerical, (2 * weights * point)[param_slice], atol=1e-5)
    assert numerical.shape == point[param_slice].shape
    assert_array_equal(point, original)
    assert len(evaluations) == 2 * numerical.size
    for offset, index in enumerate(range(param_slice.start, param_slice.stop)):
        for values, sign in zip(evaluations[2 * offset:2 * offset + 2], [-1, 1]):
            expected = point.copy()
            expected[index] += sign * 1e-4 * max(1.0, abs(point[index]))
            assert_array_equal(values, expected)


def test_finite_difference_propagates_evaluation_error():
    point = np.ones(2)
    function = Mock(side_effect=RuntimeError("evaluation failed"))
    with pytest.raises(RuntimeError, match="evaluation failed"):
        finite_difference_gradient(function, point, slice(0, 2), 1e-4)
    assert_array_equal(point, np.ones(2))


@pytest.mark.parametrize("first, second, expected", [
    ([3.0, 4.0], [3.0, 4.0], (0.0, 0.0)),
    ([3.0, 4.0], [6.0, 8.0], (0.5, 0.0)),
    ([1.0, 0.0], [-1.0, 0.0], (0.0, 2.0)),
    ([1.0, 0.0], [0.0, 1.0], (0.0, np.sqrt(2))),
    ([0.0, 0.0], [1.0, 0.0], (1.0, 1.0)),
    ([0.0, 0.0], [0.0, 0.0], (0.0, 0.0)),
])
def test_vector_error_metrics(first, second, expected):
    first, second = np.array(first), np.array(second)
    assert_allclose(vector_error_metrics(first, second), expected)
    assert_allclose(vector_error_metrics(second, first), expected)


def test_estimate_gradient_errors_refines_step():
    point = np.array([0.5])
    analytic = np.exp(point)
    initial = finite_difference_gradient(lambda values: np.exp(values[0]), point, slice(0, 1), 0.1)
    initial_mag, _ = vector_error_metrics(analytic, initial)

    magnitude, direction = estimate_gradient_errors(
        lambda values: np.exp(values[0]), point, analytic, slice(0, 1), 0.1,
        max_iterations=6,
    )

    assert magnitude < initial_mag / 100
    assert direction == pytest.approx(0.0)


@pytest.mark.parametrize("worse", [(0.2, 0.01), (0.01, 0.2)])
def test_estimate_gradient_errors_stops_when_either_metric_worsens(monkeypatch, worse):
    differences = Mock(return_value=np.ones(2))
    metrics = Mock(side_effect=[(0.1, 0.1), (0.05, 0.05), worse])
    monkeypatch.setattr("midas.validation.gradients.finite_difference_gradient", differences)
    monkeypatch.setattr("midas.validation.gradients.vector_error_metrics", metrics)

    errors = estimate_gradient_errors(Mock(), np.ones(2), np.ones(2), slice(0, 2), 0.1)

    assert errors == (0.05, 0.05)
    assert differences.call_count == 3
    assert [
        call.args[-1] for call in differences.call_args_list
    ] == pytest.approx([0.1, 0.05, 0.025])


def test_estimate_gradient_errors_iteration_limit(monkeypatch):
    differences = Mock(return_value=np.ones(2))
    monkeypatch.setattr("midas.validation.gradients.finite_difference_gradient", differences)
    errors = estimate_gradient_errors(Mock(), np.ones(2), np.ones(2), slice(0, 2), 0.1, max_iterations=0)
    assert errors == (0.0, 0.0)
    differences.assert_called_once()


@pytest.fixture
def sampled_validator(monkeypatch):
    points = np.array([[0.2, 0.3], [0.5, 0.6], [0.7, 0.8]])
    bounds = np.array([[-2.0, 4.0], [10.0, 30.0]])
    state = SimpleNamespace(
        n_params=2, parameter_names=("density", "temperature"),
        slices={"density": slice(0, 1), "temperature": slice(1, 2)},
        components=[SimpleNamespace(name="diagnostic"), SimpleNamespace(name="prior")],
    )
    normalised = Mock()
    normalised.cost.return_value = 1.0
    normalised.component_cost_gradient.return_value = np.array([2.0, 3.0])
    factory = Mock(return_value=normalised)
    rng = Mock()
    rng.uniform.return_value = points
    monkeypatch.setattr("midas.validation.gradients.PlasmaState", state)
    monkeypatch.setattr("midas.validation.gradients.NormalisedCost", factory)
    monkeypatch.setattr("midas.validation.gradients.default_rng", lambda: rng)
    return SimpleNamespace(points=points, bounds=bounds, norm=normalised, factory=factory, rng=rng)


@pytest.mark.parametrize("bad_errors, passes", [
    ([(0.01, 0.01)] * 3, 3),
    ([(0.01, 0.01), (0.2, 0.01), (0.01, 0.3)], 1),
    ([(0.1, 0.01), (0.01, 0.2), (0.4, 0.3)], 0),
])
def test_sampled_validator_aggregates_errors(sampled_validator, monkeypatch, bad_errors, passes):
    estimates = Mock(side_effect=[
        errors
        for bad in bad_errors
        for errors in [(0.01, 0.02), (0.01, 0.02), bad, (0.01, 0.02)]
    ])
    monkeypatch.setattr("midas.validation.gradients.estimate_gradient_errors", estimates)
    setup = sampled_validator
    report = validate_gradient(
        setup.bounds,
        n_samples=3,
        mag_tol=0.1,
        dir_tol=0.2,
        initial_step=1e-3,
    )

    setup.factory.assert_called_once_with(bounds=setup.bounds)
    setup.rng.uniform.assert_called_once_with(low=0.1, high=0.9, size=(3, 2))
    assert report.n_samples == 3
    assert report.success == (passes == 3)
    assert report.results["prior"]["density"] == {
        "max_mag_err": max(errors[0] for errors in bad_errors),
        "max_dir_err": max(errors[1] for errors in bad_errors), "n_passes": passes,
    }
    for component, parameters in report.results.items():
        assert set(parameters) == {"density", "temperature"}
        for name, result in parameters.items():
            if (component, name) != ("prior", "density"):
                assert result == {"max_mag_err": 0.01, "max_dir_err": 0.02, "n_passes": 3}
    assert estimates.call_count == 12
    for index, call in enumerate(estimates.call_args_list):
        assert_array_equal(call.kwargs["x0"], setup.points[index // 4])
        assert_array_equal(call.kwargs["analytic"], [2.0 if index % 2 == 0 else 3.0])
        assert call.kwargs["initial_step"] == 1e-3
    assert setup.norm.component_cost_gradient.call_count == 6
    for index, call in enumerate(setup.norm.component_cost_gradient.call_args_list):
        assert_array_equal(call.args[0], setup.points[index // 2])
        assert call.args[1] == ("diagnostic" if index % 2 == 0 else "prior")


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_sampled_validator_rejects_nonfinite_cost(sampled_validator, monkeypatch, value):
    sampled_validator.norm.cost.side_effect = [1.0, value]
    estimates = Mock()
    monkeypatch.setattr("midas.validation.gradients.estimate_gradient_errors", estimates)
    with pytest.raises(ValueError, match="non-finite costs"):
        validate_gradient(sampled_validator.bounds, n_samples=3)
    estimates.assert_not_called()


def test_gradient_with_diagnostic_and_field_prior(monkeypatch):
    monkeypatch.setattr("midas.validation.gradients.default_rng", lambda: np.random.default_rng(777))
    axis, data, sigma = StraightLine.testing_data()
    diagnostic = DiagnosticLikelihood(
        StraightLine(axis), GaussianLikelihood(y_data=data, sigma=sigma), name="line"
    )
    field_axis = np.linspace(0, 1, 3)
    field = PiecewiseLinearField(field_name="emission", axis=field_axis, axis_name="radius")
    prior = GaussianPrior(
        name="emission_prior", mean=np.zeros(3), standard_deviation=np.ones(3),
        field_request=FieldRequest(name="emission", coordinates={"radius": field_axis}),
    )
    PlasmaState.build_posterior([diagnostic], [prior], [field])
    bounds = np.array([[0.1, 3.0]] * 3 + [[1.0, 5.0], [-3.0, 0.0]])

    report = validate_gradient(bounds, n_samples=3)

    assert report.n_samples == 3
    assert set(report.results) == {"line", "emission_prior"}
    for parameters in report.results.values():
        assert set(parameters) == set(PlasmaState.slices)
    for component, parameter in [
        ("line", "gradient"), ("line", "y_intercept"),
        ("emission_prior", "emission_linear_basis"),
    ]:
        accuracy = report.results[component][parameter]
        assert accuracy["n_passes"] == 3
        assert accuracy["max_dir_err"] < 1e-4
        assert accuracy["max_mag_err"] < 1e-4


@pytest.mark.parametrize("bounds", [
    np.zeros(2), np.zeros((2, 1)), np.ones((2, 2)),
    np.array([[2.0, 1.0], [0.0, 1.0]]), np.array([[0.0, np.inf], [0.0, 1.0]]),
])
def test_invalid_parameter_bounds(monkeypatch, bounds):
    monkeypatch.setattr(PlasmaState, "n_params", 2, raising=False)
    with pytest.raises(AssertionError):
        validate_gradient(bounds)