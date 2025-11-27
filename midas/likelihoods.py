from abc import ABC, abstractmethod
from numpy import ndarray, log, exp, logaddexp, sqrt, pi, zeros, isfinite
from midas.state import PlasmaState
from midas.models import DiagnosticModel


class LikelihoodFunction(ABC):
    """
    An abstract base-class for likelihood function.
    """

    @abstractmethod
    def log_likelihood(self, predictions: ndarray) -> float:
        """
        :param predictions: \
            The model predictions of the measured data as a 1D array.

        :return: \
            The calculated log-likelihood.
        """
        pass

    @abstractmethod
    def predictions_derivative(self, predictions: ndarray) -> ndarray:
        """
        :param predictions: \
            The model predictions of the measured data as a 1D array.

        :return: \
            The derivative of the log-likelihood with respect to each element of
            ``predictions`` as a 1D array.
        """
        pass


class DiagnosticLikelihood:
    """
    A class enabling the calculation of the likelihood (and its derivative) for the data
    of a particular diagnostic.

    :param diagnostic_model: \
        An instance of a diagnostic model which inherits from the ``DiagnosticModel``
        base class.

    :param likelihood: \
        An instance of a likelihood class which inherits from the ``LikelihoodFunction``
        base class.

    :param name: \
        A name or other identifier for the diagnostic as a string.

    """

    def __init__(
        self,
        diagnostic_model: DiagnosticModel,
        likelihood: LikelihoodFunction,
        name: str,
    ):
        self.forward_model = diagnostic_model
        self.likelihood = likelihood
        self.name = name
        self.field_requests = self.forward_model.field_requests
        self.parameters = self.forward_model.parameters

    def log_probability(self) -> float:
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        predictions = self.forward_model.predictions(**param_values, **field_values)

        return self.likelihood.log_likelihood(predictions)

    def log_probability_gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = (
            PlasmaState.get_values_and_jacobians(
                parameters=self.parameters, field_requests=self.field_requests
            )
        )

        predictions, model_jacobians = self.forward_model.predictions_and_jacobians(
            **param_values, **field_values
        )

        dL_dp = self.likelihood.predictions_derivative(predictions)

        grad = zeros(PlasmaState.n_params)
        for p in param_values.keys():
            slc = PlasmaState.slices[p]
            grad[slc] = dL_dp @ model_jacobians[p]

        for field_param in field_jacobians.keys():
            field_name = PlasmaState.field_parameter_map[field_param]
            slc = PlasmaState.slices[field_param]
            grad[slc] = (dL_dp @ model_jacobians[field_name]) @ field_jacobians[
                field_param
            ]

        return grad

    def get_predictions(self):
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        return self.forward_model.predictions(**param_values, **field_values)


class GaussianLikelihood(LikelihoodFunction):
    """
    A class for constructing a Gaussian likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(self, y_data: ndarray, sigma: ndarray):
        self.y = y_data
        self.sigma = sigma

        validate_likelihood_data(
            values=y_data, uncertainties=sigma, likelihood_name=self.__class__.__name__
        )

        self.n_data = self.y.size
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_data

    def log_likelihood(self, predictions: ndarray) -> float:
        z = (self.y - predictions) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def predictions_derivative(self, predictions: ndarray) -> ndarray:
        return (self.y - predictions) * self.inv_sigma_sqr


class LogisticLikelihood(LikelihoodFunction):
    """
    A class for constructing a Logistic likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(self, y_data: ndarray, sigma: ndarray):
        self.y = y_data
        self.sigma = sigma

        validate_likelihood_data(
            values=y_data, uncertainties=sigma, likelihood_name=self.__class__.__name__
        )

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.scale = self.sigma * (sqrt(3) / pi)
        self.inv_scale = 1.0 / self.scale
        self.normalisation = -log(self.scale).sum()

    def log_likelihood(self, predictions: ndarray) -> float:
        z = (self.y - predictions) * self.inv_scale
        return z.sum() - 2 * logaddexp(0.0, z).sum() + self.normalisation

    def predictions_derivative(self, predictions: ndarray) -> ndarray:
        z = (self.y - predictions) * self.inv_scale
        return (2 / (1 + exp(-z)) - 1) * self.inv_scale


class CauchyLikelihood(LikelihoodFunction):
    """
    A class for constructing a Cauchy likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param gamma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(self, y_data: ndarray, gamma: ndarray):
        self.y = y_data
        self.gamma = gamma

        validate_likelihood_data(
            values=y_data, uncertainties=gamma, likelihood_name=self.__class__.__name__
        )

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size
        self.inv_gamma = 1.0 / self.gamma
        self.normalisation = -log(pi * self.gamma).sum()

    def log_likelihood(self, predictions: ndarray) -> float:
        z = (self.y - predictions) * self.inv_gamma
        return -log(1 + z**2).sum() + self.normalisation

    def predictions_derivative(self, predictions: ndarray) -> ndarray:
        z = (self.y - predictions) * self.inv_gamma
        return (2 * self.inv_gamma) * z / (1 + z**2)


def validate_likelihood_data(
    values: ndarray, uncertainties: ndarray, likelihood_name: str
):
    valid_types = isinstance(values, ndarray) and isinstance(uncertainties, ndarray)
    if not valid_types:
        raise TypeError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The data values and uncertainties must be instances of numpy.ndarray.
            \r>> Instead, the given types were
            \r>> {type(values)}
            \r>> and
            \r>> {type(uncertainties)}
            """
        )

    valid_shapes = (
        values.ndim == 1
        and uncertainties.ndim == 1
        and values.size == uncertainties.size
    )
    if not valid_shapes:
        raise ValueError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The data values and uncertainties arrays must be one-dimensional
            \r>> and of equal size, but instead have shapes
            \r>> {values.shape} and {uncertainties.shape}.
            """
        )

    valid_values = (
        isfinite(values).all()
        and isfinite(uncertainties).all()
        and (uncertainties > 0.0).all()
    )
    if not valid_values:
        raise ValueError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The data values and uncertainties arrays must contain only finite
            \r>> values, and all uncertainties must have values greater than zero.
            """
        )
