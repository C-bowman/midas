import pytest
from numpy import array, nan
from scipy.optimize import minimize, approx_fprime

from midas.likelihoods import GaussianLikelihood, LogisticLikelihood, CauchyLikelihood
from midas.likelihoods import ConstantUncertainty, LinearUncertainty
from midas.likelihoods import DiagnosticLikelihood
from midas import posterior, PlasmaState

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

        line_likelihood = DiagnosticLikelihood(
            likelihood=likelihood_func, diagnostic_model=model, name="straight_line"
        )

        PlasmaState.build_posterior(
            diagnostics=[line_likelihood], priors=[], field_models=[]
        )

        test_params = {
            "gradient": 1.0,
            "y_intercept": -1.0,
            "constant_error": 0.5,
            "test_constant_error": 0.3,
            "test_fractional_error": 0.05,
        }
        test_point = PlasmaState.merge_parameters(test_params)

        opt_result = minimize(
            fun=posterior.cost, x0=test_point, jac=posterior.cost_gradient
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
