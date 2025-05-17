from numpy import ndarray
from midas.likelihoods import DiagnosticLikelihood
from midas.state import PlasmaState


class Posterior:
    @staticmethod
    def log_probability(theta: ndarray) -> float:
        PlasmaState.theta = theta.copy()
        return sum(comp.log_probability() for comp in PlasmaState.components)

    @staticmethod
    def gradient(theta: ndarray) -> ndarray:
        PlasmaState.theta = theta.copy()
        return sum(comp.log_probability_gradient() for comp in PlasmaState.components)

    @staticmethod
    def cost(theta: ndarray) -> float:
        return -Posterior.log_probability(theta)

    @staticmethod
    def cost_gradient(theta: ndarray) -> ndarray:
        return -Posterior.gradient(theta)

    @staticmethod
    def component_log_probabilities(theta: ndarray) -> dict[str, float]:
        PlasmaState.theta = theta.copy()
        return {comp.name: comp.log_probability() for comp in PlasmaState.components}

    @staticmethod
    def get_model_predictions(theta: ndarray) -> dict[str, ndarray]:
        PlasmaState.theta = theta.copy()
        return {
            comp.name: comp.get_predictions() for comp in PlasmaState.components
            if isinstance(comp, DiagnosticLikelihood)
        }
