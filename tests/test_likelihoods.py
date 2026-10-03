import pytest
from numpy import array, inf, nan
from scipy.optimize import minimize, approx_fprime

from midas.likelihoods import GaussianLikelihood, LogisticLikelihood, CauchyLikelihood
from midas.likelihoods import ConstantUncertainty, LinearUncertainty, UncertaintyModel
from midas import Diagnostic, Parameters, build_posterior

from utilities import StraightLine


@pytest.mark.parametrize(
    "likelihood",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
def test_likelihood_validation(likelihood):
    y = array([1.0, 3.0, 4.0])
    sig = array([5.0, 5.0, 3.0])

    # check the type validation
    with pytest.raises(TypeError):
        likelihood(y, [s for s in sig])

    # check array shape validation
    with pytest.raises(ValueError):
        likelihood(y[:-1], sig)

    with pytest.raises(ValueError):
        likelihood(y, sig.reshape([3, 1]))

    # check finite values validation
    y[1] = nan
    with pytest.raises(ValueError):
        likelihood(y, sig)


@pytest.mark.parametrize(
    "likelihood",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
def test_likelihoods_predictions_gradient(likelihood):
    test_values = array([3.58, 2.11, 7.89])
    y = array([1.0, 3.0, 4.0])
    sig = array([5.0, 5.0, 3.0])
    func = likelihood(y, sig)

    analytic_grad, _ = func.derivatives(predictions=test_values)
    numeric_grad = approx_fprime(f=func.log_likelihood, xk=test_values)
    max_abs_err = abs(analytic_grad - numeric_grad).max()
    assert max_abs_err < 1e-6


class VectorUncertainty(UncertaintyModel):
    def __init__(self):
        self.name = "vector_uncertainty"
        self.parameters = Parameters((self.name, 2))
        self.jacobian = array([
            [1.0, 0.2],
            [0.5, 1.0],
            [1.5, 0.4],
        ])

    def get_uncertainties(self, parameters):
        return self.jacobian @ parameters[self.name]

    def get_uncertainties_and_jacobians(self, parameters):
        return self.get_uncertainties(parameters), {self.name: self.jacobian}


class PullbackVectorUncertainty(VectorUncertainty):
    def get_uncertainties_and_jacobians(self, parameters):
        raise AssertionError("The custom pullback should bypass Jacobian construction")

    def get_uncertainties_and_pullback(self, parameters):
        uncertainties = self.get_uncertainties(parameters)

        def pullback(vector):
            return {self.name: self.jacobian.T @ vector}

        return uncertainties, pullback


@pytest.mark.parametrize(
    "likelihood_function",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
@pytest.mark.parametrize("custom_pullback", [False, True])
def test_vector_parameterised_uncertainty_gradient(
    likelihood_function, custom_pullback
):
    predictions = array([0.8, 2.5, 3.7])
    y = array([1.0, 3.0, 4.0])
    parameters = array([1.2, 0.8])
    uncertainty_model = (
        PullbackVectorUncertainty() if custom_pullback else VectorUncertainty()
    )
    likelihood = likelihood_function(y, uncertainty_model)

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
    "likelihood_function",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
def test_parameterised_uncertainties(likelihood_function):
    x, y, sigma = StraightLine.testing_data()

    def run_uncertainty_model(uncertainty_model):
        likelihood_func = likelihood_function(
            y, uncertainty_model,
        )

        model = StraightLine(x_axis=x)

        line_diagnostic = Diagnostic(
            likelihood=likelihood_func, diagnostic_model=model, name="straight_line"
        )

        posterior = build_posterior(
            diagnostics=[line_diagnostic], priors=[], field_models=[]
        )

        test_params = {
            "gradient": 1.0,
            "y_intercept": -1.0,
            "constant_error": 0.5,
            "test_constant_error": 0.3,
            "test_fractional_error": 0.05,
        }
        test_point = posterior.merge_parameters(test_params)

        parameter_bounds = {
            name: (-inf, inf) for name in posterior.parameter_set
        }
        parameter_bounds.update(
            {parameter.name: (1e-3, 10.0) for parameter in uncertainty_model.parameters}
        )
        bounds = posterior.build_bounds(parameter_bounds)

        opt_result = minimize(
            fun=posterior.cost,
            x0=test_point,
            jac=posterior.cost_gradient,
            bounds=bounds,
        )

        num_grad = approx_fprime(
            xk=test_point,
            f=posterior.log_probability,
            epsilon=1e-8,
        )
        analytic_grad = posterior.gradient(test_point)

        assert abs(analytic_grad / num_grad - 1).max() < 1e-5

    constant_uncertainty = ConstantUncertainty(
        n_data=y.size, parameter_name="constant_error"
    )

    run_uncertainty_model(constant_uncertainty)

    linear_uncertainty = LinearUncertainty(y_data=y, parameter_prefix="test")

    run_uncertainty_model(linear_uncertainty)
