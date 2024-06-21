from numpy import ndarray
from midas.likelihoods import DiagnosticLikelihood
from midas.priors import BasePrior
from midas.state import PlasmaState


class Posterior:
    def __init__(self, components: list[DiagnosticLikelihood | BasePrior]):
        self.components = components
        PlasmaState.build_parametrisation(components)

    def __call__(self, theta: ndarray) -> float:
        PlasmaState.theta = theta.copy()
        return sum(comp.log_probability() for comp in self.components)

    def gradient(self, theta: ndarray) -> ndarray:
        PlasmaState.theta = theta.copy()
        return sum(comp.log_probability_gradient() for comp in self.components)

    def cost(self, theta: ndarray) -> float:
        return -self.__call__(theta)

    def cost_gradient(self, theta: ndarray) -> ndarray:
        return -self.gradient(theta)
