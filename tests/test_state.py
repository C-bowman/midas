from typing import cast

import pytest
from numpy import array, column_stack, exp, eye, linspace, ones
from numpy.random import default_rng
from numpy.testing import assert_allclose
from utilities import Polynomial, StraightLine
from midas.likelihoods import GaussianLikelihood, DiagnosticLikelihood
from midas.likelihoods.uncertainties import ConstantUncertainty
from midas.models import DiagnosticModel
from midas.models.fields import FieldModel, PiecewiseLinearField
from midas.priors import GaussianPrior
from midas.state import BasePrior, LikelihoodFunction
from midas import FieldRequest, Fields, Parameters, build_posterior


def build_diagnostic(name):
    x, y, sigma = StraightLine.testing_data()
    return DiagnosticLikelihood(
        diagnostic_model=StraightLine(x_axis=x),
        likelihood=GaussianLikelihood(y_data=y, sigma=sigma),
        name=name,
    )


@pytest.mark.parametrize("name", ["", None, 42])
def test_build_posterior_rejects_invalid_component_names(name):
    diagnostic = build_diagnostic(name)

    with pytest.raises(ValueError, match="non-empty string 'name'"):
        build_posterior(
            diagnostics=[diagnostic],
            priors=[],
            field_models=[],
        )


def test_build_posterior_rejects_duplicate_component_names():
    diagnostics = [build_diagnostic("duplicate"), build_diagnostic("duplicate")]

    with pytest.raises(ValueError, match="unique name"):
        build_posterior(
            diagnostics=diagnostics,
            priors=[],
            field_models=[],
        )


def test_diagnostic_likelihood_rejects_invalid_likelihood_type():
    x, _, _ = StraightLine.testing_data()

    with pytest.raises(TypeError, match="'likelihood' argument"):
        DiagnosticLikelihood(
            diagnostic_model=StraightLine(x_axis=x),
            likelihood=cast(LikelihoodFunction, object()), # assign an invalid type to 'likelihood'
            name="invalid_likelihood",
        )


def test_diagnostic_likelihood_rejects_invalid_likelihood_parameters():
    x, y, sigma = StraightLine.testing_data()
    likelihood = GaussianLikelihood(y_data=y, sigma=sigma)
    # assign an invalid type to 'parameters'
    likelihood.parameters = cast(Parameters, [])

    with pytest.raises(TypeError, match="valid 'parameters' instance attribute"):
        DiagnosticLikelihood(
            diagnostic_model=StraightLine(x_axis=x),
            likelihood=likelihood,
            name="invalid_likelihood_parameters",
        )


def build_field_model(field_name="emission"):
    return PiecewiseLinearField(
        field_name=field_name,
        axis=linspace(0, 1, 3),
        axis_name="radius",
    )


@pytest.mark.parametrize("name", ["", None, 42])
def test_build_posterior_rejects_invalid_field_model_names(name):
    field_model = build_field_model()
    field_model.name = cast(str, name)

    with pytest.raises(ValueError, match="field model must have a non-empty string"):
        build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[field_model],
        )


def test_build_posterior_rejects_invalid_field_model_parameters():
    field_model = build_field_model()
    field_model.parameters = cast(Parameters, [])

    with pytest.raises(TypeError, match="valid 'parameters' instance attribute"):
        build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[field_model],
        )


@pytest.mark.parametrize("parameter_size", [2, 3])
def test_build_posterior_with_shared_field_parameters(parameter_size):
    field_models = [build_field_model("emission"), build_field_model("temperature")]
    priors = []
    for model, size in zip(field_models, [3, parameter_size]):
        model.param_name = "shared_parameter"
        model.parameters = Parameters(("shared_parameter", size))
        priors.append(GaussianPrior(
            name=f"{model.name}_prior",
            mean=array([0.0, 0.0, 0.0]),
            standard_deviation=array([1.0, 1.0, 1.0]),
            field_request=FieldRequest(model.name, {"radius": linspace(0, 1, 3)}),
        ))

    if parameter_size != 3:
        with pytest.raises(ValueError, match="differ in their size"):
            build_posterior([], priors, field_models)
    else:
        posterior = build_posterior([], priors, field_models)
        assert posterior.slices == {"shared_parameter": slice(0, 3)}
        assert posterior.parameter_sizes == {"shared_parameter": 3}
        assert posterior.n_params == 3


def test_build_posterior_rejects_duplicate_field_names():
    with pytest.raises(ValueError, match="unique field name"):
        build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[build_field_model(), build_field_model()],
        )


def test_build_posterior_rejects_zero_parameter_posterior():
    with pytest.raises(ValueError, match="must contain at least one parameter"):
        build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[],
        )


def test_build_posterior_generates_consistent_parameter_mappings():
    diagnostic = build_diagnostic("line")
    field_model = build_field_model()
    field_request = FieldRequest(
        name="emission",
        coordinates={"radius": linspace(0, 1, 3)},
    )
    prior = GaussianPrior(
        name="emission_prior",
        mean=array([0.0, 0.0, 0.0]),
        standard_deviation=array([1.0, 1.0, 1.0]),
        field_request=field_request,
    )

    posterior = build_posterior(
        diagnostics=[diagnostic],
        priors=[prior],
        field_models=[field_model],
    )

    assert posterior.slices == {
        "emission_linear_basis": slice(0, 3),
        "gradient": slice(3, 4),
        "y_intercept": slice(4, 5),
    }
    assert posterior.parameter_names == (
        "emission_linear_basis",
        "gradient",
        "y_intercept",
    )
    assert posterior.parameter_set == set(posterior.slices)
    assert posterior.parameter_sizes == {
        "emission_linear_basis": 3,
        "gradient": 1,
        "y_intercept": 1,
    }
    assert posterior.n_params == 5


def test_build_bounds():
    x, y, sigma = StraightLine.testing_data()

    poly_model = Polynomial(x_axis=x, order=2)
    likelihood = GaussianLikelihood(y_data=y, sigma=sigma)
    diagnostic = DiagnosticLikelihood(
        diagnostic_model=poly_model,
        likelihood=likelihood,
        name="poly"
    )

    posterior = build_posterior(
        diagnostics=[diagnostic],
        priors=[],
        field_models=[]
    )

    # first test that we can assign all parameters the same bounds with a tuple
    param_bounds = {
        "poly_coefficients": (-10.0, 10.0),
    }
    bounds = posterior.build_bounds(param_bounds)

    # now test we can assign different bounds using an array of the correct shape
    param_bounds = {
        "poly_coefficients": array([(-1, 1), (-2, 2), (-3, 3)]),
    }
    bounds = posterior.build_bounds(param_bounds)


class CoupledField(FieldModel):
    def __init__(self, name, scalar_matrix):
        self.name = name
        self.n_params = 4
        self.offset_name = f"{name}_offset"
        self.scalar_matrix = scalar_matrix
        self.parameters = Parameters(("shared", 2), ("scale", 1), (self.offset_name, 1))

    def get_values(self, parameters, field):
        radius = field.coordinates["radius"]
        basis = column_stack((ones(radius.size), radius))
        return exp(
            basis @ parameters["shared"]
            + radius**2 * parameters["scale"]
            + parameters[self.offset_name]
        )

    def get_values_and_jacobian(self, parameters, field):
        radius = field.coordinates["radius"]
        values = self.get_values(parameters, field)
        basis = column_stack((ones(radius.size), radius))
        scale_jacobian = values * radius**2
        offset_jacobian = values.copy()
        if self.scalar_matrix:
            scale_jacobian = scale_jacobian[:, None]
            offset_jacobian = offset_jacobian[:, None]
        return values, {
            "shared": values[:, None] * basis,
            "scale": scale_jacobian,
            self.offset_name: offset_jacobian,
        }


class CoupledDiagnostic(DiagnosticModel):
    def __init__(self, fields):
        self.fields = Fields(*fields)
        self.parameters = Parameters(("shared", 2), ("scale", 1), ("bias", 1))
        self.jacobians = {
            "shared": array([[0.3, -0.2], [0.1, 0.4]]),
            "scale": array([0.5, -0.3]),
            "bias": ones(2),
        }
        for field in fields:
            self.jacobians[field.name] = linspace(-0.2, 0.5, 2 * field.size).reshape(
                2, field.size
            )

    def predictions(self, **values):
        result = self.jacobians["shared"] @ values["shared"]
        result += self.jacobians["scale"] * values["scale"] + values["bias"]
        for field in self.fields:
            result += self.jacobians[field.name] @ values[field.name]
        return result

    def predictions_and_jacobians(self, **values):
        return self.predictions(**values), self.jacobians


class CoupledPrior(BasePrior):
    def __init__(self, fields):
        self.name = "coupled_prior"
        self.fields = Fields(*fields)
        self.parameters = Parameters(("shared", 2), ("scale", 1))

    def probability(self, **values):
        return -0.5 * sum((value**2).sum() for value in values.values())

    def gradients(self, **values):
        return {name: -value for name, value in values.items()}


def build_coupled_posterior(scalar_matrix=False, reverse=False):
    requests = [
        FieldRequest("emission", {"radius": linspace(0, 0.6, 3)}),
        FieldRequest("temperature", {"radius": linspace(0.1, 0.7, 4)}),
    ]
    fields = [CoupledField(request.name, scalar_matrix) for request in requests]
    if reverse:
        fields.reverse()
        requests.reverse()
    diagnostics = [
        DiagnosticLikelihood(
            diagnostic_model=CoupledDiagnostic(requested),
            likelihood=GaussianLikelihood(
                y_data=array([0.7, -0.2]),
                sigma=ConstantUncertainty(n_data=2, parameter_name="scale"),
            ),
            name=f"diagnostic_{index}",
        )
        for index, requested in enumerate([
            requests,
            [FieldRequest("emission", {"radius": linspace(-0.1, 0.4, 5)})],
            [FieldRequest("temperature", {"radius": linspace(0.2, 0.8, 2)})],
            [],
        ])
    ]
    return build_posterior(
        diagnostics=diagnostics,
        priors=[CoupledPrior(requests)],
        field_models=fields,
    )


def finite_difference(function, theta):
    step = 1e-5
    directions = eye(theta.size) * step
    return array([
        (function(theta + direction) - function(theta - direction)) / (2 * step)
        for direction in directions
    ])


@pytest.mark.parametrize("scalar_matrix", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("seed", [1, 7, 23])
def test_shared_parameter_gradients(scalar_matrix, reverse, seed):
    posterior = build_coupled_posterior(scalar_matrix, reverse)
    assert posterior.n_params == 6
    assert posterior.parameter_sizes == {
        "bias": 1,
        "emission_offset": 1,
        "scale": 1,
        "shared": 2,
        "temperature_offset": 1,
    }
    rng = default_rng(seed)
    theta = rng.uniform(-0.3, 0.3, posterior.n_params)
    theta[posterior.slices["scale"]] = 1.2
    for component in posterior.components:
        def probability_at(point):
            return posterior.component_log_probability(point, component.name)

        numerical = finite_difference(probability_at, theta)
        assert_allclose(
            posterior.component_gradient(theta, component.name),
            numerical,
            rtol=1e-7,
            atol=1e-8,
        )
    numerical = finite_difference(posterior.log_probability, theta)
    assert_allclose(posterior.gradient(theta), numerical, rtol=1e-7, atol=1e-8)


def test_shared_parameter_jacobians_are_grouped_by_field():
    posterior = build_coupled_posterior()
    context = posterior._context(ones(posterior.n_params) * 0.2)
    diagnostic = posterior.components[0]
    parameters, values, jacobians = context.get_values_and_jacobians(
        diagnostic.model_parameters, diagnostic.fields
    )
    assert set(parameters) == {"shared", "scale", "bias"}
    assert set(values) == set(jacobians) == {"emission", "temperature"}
    for request in diagnostic.fields:
        model = posterior.field_models[request.name]
        expected_values, expected_jacobians = model.get_values_and_jacobian(
            context.get_parameter_values(model.parameters), request
        )
        assert_allclose(values[request.name], expected_values)
        assert set(jacobians[request.name]) == set(expected_jacobians)
        for param_name, expected in expected_jacobians.items():
            assert_allclose(jacobians[request.name][param_name], expected)
