from typing import cast

import pytest
from numpy import array, linspace
from utilities import Polynomial, StraightLine
from midas.likelihoods import GaussianLikelihood, DiagnosticLikelihood
from midas.models.fields import PiecewiseLinearField
from midas.priors import GaussianPrior
from midas.state import LikelihoodFunction
from midas import FieldRequest, Parameters, PlasmaState


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
        PlasmaState.build_posterior(
            diagnostics=[diagnostic],
            priors=[],
            field_models=[],
        )


def test_build_posterior_rejects_duplicate_component_names():
    diagnostics = [build_diagnostic("duplicate"), build_diagnostic("duplicate")]

    with pytest.raises(ValueError, match="unique name"):
        PlasmaState.build_posterior(
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
        PlasmaState.build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[field_model],
        )


def test_build_posterior_rejects_invalid_field_model_parameters():
    field_model = build_field_model()
    field_model.parameters = cast(Parameters, [])

    with pytest.raises(TypeError, match="valid 'parameters' instance attribute"):
        PlasmaState.build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[field_model],
        )


def test_build_posterior_rejects_field_parameter_owned_by_multiple_models():
    emission_model = build_field_model("emission")
    temperature_model = build_field_model("temperature")
    emission_model.parameters = Parameters(("shared_parameter", 3))
    temperature_model.parameters = Parameters(("shared_parameter", 3))

    with pytest.raises(ValueError, match="belong to only one field model"):
        PlasmaState.build_posterior(
            diagnostics=[],
            priors=[],
            field_models=[emission_model, temperature_model],
        )


def test_build_posterior_rejects_zero_parameter_posterior():
    with pytest.raises(ValueError, match="must contain at least one parameter"):
        PlasmaState.build_posterior(
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

    PlasmaState.build_posterior(
        diagnostics=[diagnostic],
        priors=[prior],
        field_models=[field_model],
    )

    assert PlasmaState.slices == {
        "emission_linear_basis": slice(0, 3),
        "gradient": slice(3, 4),
        "y_intercept": slice(4, 5),
    }
    assert PlasmaState.parameter_names == (
        "emission_linear_basis",
        "gradient",
        "y_intercept",
    )
    assert PlasmaState.parameter_set == set(PlasmaState.slices)
    assert PlasmaState.parameter_sizes == {
        "emission_linear_basis": 3,
        "gradient": 1,
        "y_intercept": 1,
    }
    assert PlasmaState.n_params == 5
    assert PlasmaState.field_parameter_map == {
        "emission_linear_basis": "emission",
    }


def test_build_bounds():
    x, y, sigma = StraightLine.testing_data()

    poly_model = Polynomial(x_axis=x, order=2)
    likelihood = GaussianLikelihood(y_data=y, sigma=sigma)
    diagnostic = DiagnosticLikelihood(
        diagnostic_model=poly_model,
        likelihood=likelihood,
        name="poly"
    )

    PlasmaState.build_posterior(
        diagnostics=[diagnostic],
        priors=[],
        field_models=[]
    )

    # first test that we can assign all parameters the same bounds with a tuple
    param_bounds = {
        "poly_coefficients": (-10.0, 10.0),
    }
    bounds = PlasmaState.build_bounds(param_bounds)

    # now test we can assign different bounds using an array of the correct shape
    param_bounds = {
        "poly_coefficients": array([(-1, 1), (-2, 2), (-3, 3)]),
    }
    bounds = PlasmaState.build_bounds(param_bounds)
