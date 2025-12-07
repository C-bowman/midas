import pytest
from numpy import array, nan
from scipy.optimize import minimize, approx_fprime

from midas.likelihoods import GaussianLikelihood, LogisticLikelihood, CauchyLikelihood
from midas.likelihoods import ScaledGaussianLikelihood
from midas.likelihoods import DiagnosticLikelihood
from midas import posterior, PlasmaState

from utilities import StraightLine


@pytest.mark.parametrize(
    "likelihood",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
def test_likelihood_validation(likelihood):
    y = array([1., 3., 4.])
    sig = array([5., 5., 3.])

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
    y = array([1., 3., 4.])
    sig = array([5., 5., 3.])
    func = likelihood(y, sig)

    analytic_grad, _ = func.derivatives(predictions=test_values)
    numeric_grad = approx_fprime(f=func.log_likelihood, xk=test_values)
    max_abs_err = abs(analytic_grad - numeric_grad).max()
    assert max_abs_err < 1e-6


def test_scaled_gaussian():
    x, y, sigma = StraightLine.testing_data()
    likelihood_func = ScaledGaussianLikelihood(
        y_data=y,
        sigma=sigma,
        scale_parameter_name="error_scale"
    )

    model = StraightLine(x_axis=x)

    line_likelihood = DiagnosticLikelihood(
        likelihood=likelihood_func,
        diagnostic_model=model,
        name="straight_line"
    )

    PlasmaState.build_posterior(
        diagnostics=[line_likelihood],
        priors=[],
        field_models=[]
    )

    test_params = {
        "gradient": 1.0,
        "y_intercept": -1.0,
        "error_scale": 0.5
    }
    test_point = PlasmaState.merge_parameters(test_params)

    opt_result = minimize(
        fun=posterior.cost,
        x0=test_point,
        jac=posterior.cost_gradient
    )

    num_grad = approx_fprime(
        xk=test_point,
        f=posterior.log_probability,
        epsilon=1e-8,
    )
    analytic_grad = posterior.gradient(test_point)

    assert abs(analytic_grad/num_grad - 1).max() < 1e-6