from numpy import array, ndarray
from numpy.testing import assert_allclose
from scipy.optimize import minimize, approx_fprime
import pytest

from midas.likelihoods import GaussianLikelihood, UncertaintyModel
from midas.models import DiagnosticModel, FieldModel
from midas import Diagnostic, FieldRequest, Fields, Parameters, build_posterior

from utilities import StraightLine


def test_default_diagnostic_pullback_uses_jacobians():
    x = array([1.0, 2.0, 4.0])
    model = StraightLine(x)
    predictions, pullback = model.predictions_and_pullback(
        gradient=array([2.0]), y_intercept=array([0.5])
    )

    vector = array([0.2, -0.5, 1.0])
    gradients = pullback(vector)

    assert_allclose(predictions, 2.0 * x + 0.5)
    assert_allclose(gradients["gradient"], array([vector @ x]))
    assert_allclose(gradients["y_intercept"], array([vector.sum()]))
    assert gradients["gradient"].shape == (1,)
    assert gradients["y_intercept"].shape == (1,)


class CustomPullbackModel(DiagnosticModel):
    def __init__(self, matrix: ndarray):
        self.matrix = matrix
        self.parameters = Parameters(("coefficients", matrix.shape[1]))
        self.fields = Fields()

    def predictions(self, **values: ndarray) -> ndarray:
        return self.matrix @ values["coefficients"]

    def predictions_and_pullback(self, **values: ndarray):
        predictions = self.predictions(**values)

        def pullback(vector: ndarray) -> dict[str, ndarray]:
            return {"coefficients": self.matrix.T @ vector}

        return predictions, pullback


class PullbackOnlyField(FieldModel):
    def __init__(self):
        self.name = "field"
        self.parameters = Parameters()

    def values(self, parameters, field):
        return array([1.0])

    def values_and_pullback(self, parameters, field):
        return self.values(parameters, field), lambda vector: {}


class PullbackOnlyUncertainty(UncertaintyModel):
    def __init__(self):
        self.parameters = Parameters()

    def uncertainties(self, parameters):
        return array([1.0])

    def uncertainties_and_pullback(self, parameters):
        return self.uncertainties(parameters), lambda vector: {}


def test_jacobian_methods_are_optional():
    assert DiagnosticModel.__abstractmethods__ == frozenset({"predictions"})
    assert FieldModel.__abstractmethods__ == frozenset({"values"})
    assert UncertaintyModel.__abstractmethods__ == frozenset({"uncertainties"})

    diagnostic = CustomPullbackModel(array([[1.0]]))
    field = PullbackOnlyField()
    uncertainty = PullbackOnlyUncertainty()

    with pytest.raises(NotImplementedError, match="predictions_and_jacobians"):
        diagnostic.predictions_and_jacobians(coefficients=array([1.0]))
    with pytest.raises(NotImplementedError, match="values_and_jacobians"):
        field.values_and_jacobians(
            {}, FieldRequest("field", {"x": array([0.0])})
        )
    with pytest.raises(NotImplementedError, match="uncertainties_and_jacobians"):
        uncertainty.uncertainties_and_jacobians({})


def test_posterior_uses_custom_diagnostic_pullback():
    matrix = array([[1.0, 0.2], [0.5, 1.5], [-0.3, 0.8]])
    model = CustomPullbackModel(matrix)
    likelihood = GaussianLikelihood(
        y_data=array([1.0, -0.5, 0.8]),
        sigma=array([0.5, 0.7, 0.9]),
    )
    posterior = build_posterior(
        diagnostics=[Diagnostic(model, likelihood, "custom_pullback")],
        priors=[],
        field_models=[],
    )
    point = array([0.4, -0.2])

    numerical = approx_fprime(point, posterior.log_probability)

    assert_allclose(posterior.gradient(point), numerical, rtol=1e-6, atol=1e-7)

def test_straight_line_fit():
    # Here we verify that we can fit a simple straight-line model to some
    # data without specifying any fields in the problem
    x, y, sigma = StraightLine.testing_data()
    likelihood_func = GaussianLikelihood(
        y_data=y,
        sigma=sigma
    )

    model = StraightLine(x_axis=x)

    line_diagnostic = Diagnostic(
        likelihood=likelihood_func,
        diagnostic_model=model,
        name="straight_line"
    )

    posterior = build_posterior(
        diagnostics=[line_diagnostic],
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