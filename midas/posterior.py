from numpy import array, ndarray, zeros
from collections import defaultdict

from midas import FieldRequest
from midas.state import PlasmaState, DiagnosticLikelihood


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

    diagnostics = [d for d in PlasmaState.components if isinstance(d, DiagnosticLikelihood)]
    # group model predictions for each sample into lists
    for theta in parameter_samples:
        PlasmaState.theta = theta.copy()
        for diagnostic in diagnostics:
            predictions[diagnostic.name].append(diagnostic.get_predictions())

    # convert the lists of arrays into 2D arrays
    predictions = {name: array(val) for name, val in predictions.items()}
    return predictions


def sample_field_values(parameter_samples: ndarray, field_request: FieldRequest) -> ndarray:
    """
    Calculate the values of a requested field for a given set of parameter samples.

    :param parameter_samples: \
        The model parameter samples as a 2D array with shape
        ``(n_samples, n_parameters)``.

    :param field_request: \
        A ``FieldRequest`` specifying which field values should be calculated from
        the given parameter samples.

    :return: \
        The requested field values calculated for each sample as a 2D array.
    """
    assert isinstance(parameter_samples, ndarray)
    assert parameter_samples.ndim == 2
    assert parameter_samples.shape[1] == PlasmaState.n_params
    assert field_request.name in PlasmaState.field_models

    field_model = PlasmaState.field_models[field_request.name]
    field_values = zeros([parameter_samples.shape[0], field_request.size])
    # group model predictions for each sample into lists
    for i, theta in enumerate(parameter_samples):
        param_dict = PlasmaState.split_parameters(theta)
        field_values[i, :] = field_model.get_values(parameters=param_dict, field=field_request)

    return field_values