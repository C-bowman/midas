from numpy import ndarray, ones_like, linspace, zeros
from numpy.random import default_rng
from midas import Parameters, Fields
from midas.models import DiagnosticModel


class StraightLine(DiagnosticModel):
    def __init__(self, x_axis: ndarray):
        self.x = x_axis
        self.parameters = Parameters(
            ("gradient", 1),
            ("y_intercept", 1),
        )
        self.fields = Fields()
        self.intercept_jac = ones_like(self.x)

    def predictions(self, gradient: float, y_intercept: float) -> ndarray:
        return gradient * self.x + y_intercept

    def predictions_and_jacobians(self, gradient: float, y_intercept: float) -> tuple:
        predictions = gradient * self.x + y_intercept
        jacobians = {"gradient": self.x, "y_intercept": self.intercept_jac}
        return predictions, jacobians

    @staticmethod
    def testing_data():
        rng = default_rng(777)
        x = linspace(1, 10, 10)
        sigma = zeros(x.size) + 2.0
        y = 3.5 * x - 2.0 + rng.normal(size=x.size, scale=sigma)
        return x, y, sigma