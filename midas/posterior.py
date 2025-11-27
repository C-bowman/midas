from numpy import array, ndarray
from collections import defaultdict
from midas.likelihoods import DiagnosticLikelihood
from midas.state import PlasmaState


def log_probability(theta: ndarray) -> float:
    """
    Calculate the posterior log-probability for a given set of model parameters.

    :param theta: \
        The model parameter values as a 1D array.

    :return: \
        The posterior log-probability.
    """
    PlasmaState.theta = theta.copy()
    return sum(comp.log_probability() for comp in PlasmaState.components)


def gradient(theta: ndarray) -> ndarray:
    """
    Calculate the gradient of posterior log-probability with respect to the model parameters.

    :param theta: \
        The model parameter values as a 1D array.

    :return: \
        The gradient of the posterior log-probability as a 1D array.
    """
    PlasmaState.theta = theta.copy()
    return sum(comp.log_probability_gradient() for comp in PlasmaState.components)


def cost(theta: ndarray) -> float:
    """
    Calculate the 'cost' (the negative posterior log-probability)
    for a given set of model parameters.

    :param theta: \
        The model parameter values as a 1D array.

    :return: \
        The negative posterior log-probability.
    """
    return -log_probability(theta)


def cost_gradient(theta: ndarray) -> ndarray:
    """
    Calculate the gradient of the 'cost' (the negative posterior log-probability)
    with respect to the model parameters.

    :param theta: \
        The model parameter values as a 1D array.

    :return: \
        The gradient of the negative posterior log-probability as a 1D array.
    """
    return -gradient(theta)


def component_log_probabilities(theta: ndarray) -> dict[str, float]:
    """
    Calculate the log-probability of each component of the posterior (i.e. each
    individual diagnostic likelihood and prior distribution).

    :param theta: \
        The model parameter values as a 1D array.

    :return: \
        A dictionary mapping the name of each posterior component to its
        corresponding log-probability.
    """
    PlasmaState.theta = theta.copy()
    return {comp.name: comp.log_probability() for comp in PlasmaState.components}


def get_model_predictions(theta: ndarray) -> dict[str, ndarray]:
    """
    Calculate the predictions of the forward-model associated with each
    ``DiagnosticLikelihood`` component in the posterior distribution.

    :param theta: \
        The model parameter values as a 1D array.

    :return: \
        A dictionary mapping the name of each ``DiagnosticLikelihood`` to its
        corresponding forward-model predictions as a 1D array.
    """
    PlasmaState.theta = theta.copy()
    return {
        comp.name: comp.get_predictions()
        for comp in PlasmaState.components
        if isinstance(comp, DiagnosticLikelihood)
    }


def sample_model_predictions(parameter_samples: ndarray) -> dict[str, ndarray]:
    """
    Calculate the predictions of the forward-model associated with each
    ``DiagnosticLikelihood`` component in the posterior distribution.

    :param parameter_samples: \
        The model parameter samples as a 2D array with shape
        ``(n_samples, n_parameters)``.

    :return: \
        A dictionary mapping the name of each ``DiagnosticLikelihood`` to its
        corresponding forward-model predictions for each sample as a 2D array.
    """
    assert isinstance(parameter_samples, ndarray)
    assert parameter_samples.ndim == 2
    assert parameter_samples.shape[1] == PlasmaState.n_params

    predictions = defaultdict(list)

    # group model predictions for each sample into lists
    for theta in parameter_samples:
        PlasmaState.theta = theta.copy()
        for comp in PlasmaState.components:
            predictions[comp.name].append(comp.get_predictions())

    # convert the lists of arrays into 2D arrays
    predictions = {name: array(val) for name, val in predictions.items()}
    return predictions
