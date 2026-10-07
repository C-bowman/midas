import pytest
from numpy import allclose, array, nan
from scipy.optimize import approx_fprime

from midas.likelihoods import (
    GaussianLikelihood,
    SplitGaussianLikelihood,
    LogisticLikelihood,
    CauchyLikelihood,
)
from midas.likelihoods import ConstantUncertainty, LinearUncertainty, UncertaintyModel
from midas import Diagnostic, Parameters, build_posterior

from utilities import StraightLine


likelihood_test_setup = [
    (
        GaussianLikelihood,
        {"sigma": array([5.0, 5.0, 3.0])},
    ),
    (
        SplitGaussianLikelihood,
        {
            "sigma_lower": array([5.0, 5.0, 3.0]),
            "sigma_upper": array([6.0, 6.0, 4.0]),
        },
    ),
    (
        LogisticLikelihood,
        {"sigma": array([5.0, 5.0, 3.0])},
    ),
    (
        CauchyLikelihood,
        {"gamma": array([5.0, 5.0, 3.0])},
    ),
]


def constant_uncertainty_setup(y_data, index):
    parameter_name = f"constant_error_{index}"
    model = ConstantUncertainty(y_data.size, parameter_name)
    return model, {parameter_name: 0.5 + 0.1 * index}


def linear_uncertainty_setup(y_data, index):
    parameter_prefix = f"error_{index}"
    model = LinearUncertainty(y_data, parameter_prefix)
    parameters = {
        f"{parameter_prefix}_constant_error": 0.3,
        f"{parameter_prefix}_fractional_error": 0.05 + 0.01 * index,
    }
    return model, parameters


uncertainty_model_test_setup = [
    constant_uncertainty_setup,
    linear_uncertainty_setup,
]


@pytest.mark.parametrize(
    "likelihood_class, kwargs", likelihood_test_setup
)
def test_likelihood_validation(likelihood_class, kwargs):
    y = array([1.0, 3.0, 4.0])

    for argument in kwargs:
        invalid = kwargs.copy()
        invalid[argument] = invalid[argument].tolist()
        with pytest.raises(TypeError):
            likelihood_class(y_data=y, **invalid)

        invalid[argument] = kwargs[argument].reshape([3, 1])
        with pytest.raises(ValueError):
            likelihood_class(y_data=y, **invalid)

    with pytest.raises(ValueError):
        likelihood_class(y_data=y[:-1], **kwargs)

    invalid_y = y.copy()
    invalid_y[1] = nan
    with pytest.raises(ValueError):
        likelihood_class(y_data=invalid_y, **kwargs)


@pytest.mark.parametrize(
    "likelihood_class, kwargs", likelihood_test_setup
)
def test_likelihoods_predictions_gradient(likelihood_class, kwargs):
    test_values = array([3.58, 2.11, 7.89])
    y = array([1.0, 3.0, 4.0])
    likelihood = likelihood_class(y_data=y, **kwargs)

    analytic_grad, _ = likelihood.derivatives(predictions=test_values)
    numeric_grad = approx_fprime(f=likelihood.log_likelihood, xk=test_values)
    max_abs_err = abs(analytic_grad - numeric_grad).max()
    assert max_abs_err < 1e-6


def test_split_gaussian_matches_gaussian_when_uncertainties_are_equal():
    y = array([1.0, 3.0, 4.0])
    sigma = array([0.5, 1.5, 2.0])
    predictions = array([2.0, 3.0, 2.5])
    gaussian = GaussianLikelihood(y, sigma)
    split_gaussian = SplitGaussianLikelihood(y, sigma, sigma)

    assert allclose(
        split_gaussian.log_likelihood(predictions),
        gaussian.log_likelihood(predictions),
    )
    assert allclose(
        split_gaussian.derivatives(predictions)[0],
        gaussian.derivatives(predictions)[0],
    )


@pytest.mark.parametrize("model_is_lower", [True, False])
def test_split_gaussian_rejects_mixed_uncertainty_types(model_is_lower):
    y = array([1.0, 3.0, 4.0])
    fixed = array([0.5, 1.5, 2.0])
    model = ConstantUncertainty(y.size, "error")
    sigma_lower, sigma_upper = (model, fixed) if model_is_lower else (fixed, model)

    with pytest.raises(ValueError, match="must either both be arrays"):
        SplitGaussianLikelihood(y, sigma_lower, sigma_upper)


class VectorUncertainty(UncertaintyModel):
    def __init__(self, name="vector_uncertainty"):
        self.name = name
        self.parameters = Parameters((self.name, 2))
        self.jacobian = array([
            [1.0, 0.2],
            [0.5, 1.0],
            [1.5, 0.4],
        ])

    def uncertainties(self, parameters):
        return self.jacobian @ parameters[self.name]

    def uncertainties_and_jacobians(self, parameters):
        return self.uncertainties(parameters), {self.name: self.jacobian}


class PullbackVectorUncertainty(VectorUncertainty):
    def uncertainties_and_jacobians(self, parameters):
        raise AssertionError("The custom pullback should bypass Jacobian construction")

    def uncertainties_and_pullback(self, parameters):
        uncertainties = self.uncertainties(parameters)

        def pullback(vector):
            return {self.name: self.jacobian.T @ vector}

        return uncertainties, pullback


@pytest.mark.parametrize(
    "likelihood_class, kwargs", likelihood_test_setup
)
@pytest.mark.parametrize("custom_pullback", [False, True])
def test_vector_parameterised_uncertainty_gradient(
    likelihood_class, kwargs, custom_pullback
):
    predictions = array([0.8, 2.5, 3.7])
    y = array([1.0, 3.0, 4.0])
    parameters = array([1.2, 0.8])
    uncertainty_model = (
        PullbackVectorUncertainty() if custom_pullback else VectorUncertainty()
    )
    parameterised_kwargs = kwargs.copy()
    for argument in kwargs:
        parameterised_kwargs[argument] = uncertainty_model
    likelihood = likelihood_class(y_data=y, **parameterised_kwargs)

    _, derivatives = likelihood.derivatives(
        predictions, vector_uncertainty=parameters
    )
    numerical = approx_fprime(
        parameters,
        lambda values: likelihood.log_likelihood(
            predictions, vector_uncertainty=values
        ),
    )

    assert derivatives["vector_uncertainty"].shape == parameters.shape
    assert abs(derivatives["vector_uncertainty"] - numerical).max() < 1e-6


@pytest.mark.parametrize(
    "likelihood_class, kwargs", likelihood_test_setup
)
@pytest.mark.parametrize("uncertainty_setup", uncertainty_model_test_setup)
def test_parameterised_uncertainties(
    likelihood_class, kwargs, uncertainty_setup
):
    x, y, _ = StraightLine.testing_data()
    parameterised_kwargs = kwargs.copy()
    uncertainty_parameters = {}

    for index, argument in enumerate(kwargs):
        model, parameters = uncertainty_setup(y, index)
        parameterised_kwargs[argument] = model
        uncertainty_parameters.update(parameters)

    likelihood = likelihood_class(y_data=y, **parameterised_kwargs)
    diagnostic = Diagnostic(
        likelihood=likelihood,
        diagnostic_model=StraightLine(x_axis=x),
        name="straight_line",
    )
    posterior = build_posterior(
        diagnostics=[diagnostic], priors=[], field_models=[]
    )
    test_point = posterior.merge_parameters(
        {
            "gradient": 1.0,
            "y_intercept": -1.0,
            **uncertainty_parameters,
        }
    )

    numerical = approx_fprime(test_point, posterior.log_probability)
    analytic = posterior.gradient(test_point)

    assert allclose(analytic, numerical, rtol=1e-5, atol=1e-7)
