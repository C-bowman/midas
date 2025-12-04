from numpy import array, ndarray, linspace, zeros
from numpy.random import default_rng
from scipy.optimize import minimize

from midas.models import DiagnosticModel
from midas.parameters import ParameterVector, Parameters, Fields
from midas.likelihoods import GaussianLikelihood
from midas.state import PlasmaState, DiagnosticLikelihood
from midas import posterior


def straight_line_data():
    rng = default_rng()
    x = linspace(1, 10, 10)
    sigma = zeros(x.size) + 2.0
    y = 3.5 * x - 2.0 + rng.normal(size=x.size, scale=sigma)
    return x, y, sigma


class StraightLine(DiagnosticModel):
    def __init__(self, x_axis: ndarray):
        self.axis = x_axis
        self.parameters = Parameters(
            ParameterVector(name="line_coefficients", size=2)
        )
        self.fields = Fields()

    def predictions(self, line_coefficients):
        grad, offset = line_coefficients
        return grad * self.axis + offset

    def predictions_and_jacobians(self, line_coefficients):
        grad, offset = line_coefficients
        jacobian = zeros([self.axis.size, 2])
        jacobian[:, 0] = self.axis
        jacobian[:, 1] = 1.0
        return grad * self.axis + offset, {"line_coefficients": jacobian}


def test_straight_line_fit():
    # Here we verify that we can fit a simple straight-line model to some
    # data without specifying any fields in the problem
    x, y, sigma = straight_line_data()
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

    opt_result = minimize(
        fun=posterior.cost,
        x0=array([1.0, -1.0]),
        jac=posterior.cost_gradient
    )

