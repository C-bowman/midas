from numpy import ndarray, log, exp, logaddexp, sqrt, pi, isfinite
from midas.state import LikelihoodFunction, DiagnosticLikelihood


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
