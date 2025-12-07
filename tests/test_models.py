from numpy import array
from scipy.optimize import minimize, approx_fprime
from midas.likelihoods import GaussianLikelihood
from midas.state import PlasmaState, DiagnosticLikelihood
from midas import posterior

from utilities import StraightLine

def test_straight_line_fit():
    # Here we verify that we can fit a simple straight-line model to some
    # data without specifying any fields in the problem
    x, y, sigma = StraightLine.testing_data()
    likelihood_func = GaussianLikelihood(
        y_data=y,
        sigma=sigma
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

    test_point = array([1.0, -1.0])

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