from numpy import ndarray, log, exp, logaddexp, sqrt, pi, isfinite
from midas.state import LikelihoodFunction, DiagnosticLikelihood
from midas.parameters import Parameters
from midas.likelihoods.uncertainties import UncertaintyModel


class GaussianLikelihood(LikelihoodFunction):
    """
    A class for constructing a Gaussian likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(self, y_data: ndarray, sigma: ndarray | UncertaintyModel):
        self.y = y_data

        validate_likelihood_data(
            values=y_data, uncertainties=sigma, likelihood_name=self.__class__.__name__
        )

        self.n_data = self.y.size
        if isinstance(sigma, UncertaintyModel):
            self.uncertainty_model = sigma
            self.normalisation = -0.5 * log(2 * pi) * self.n_data
            self.parameters = self.uncertainty_model.parameters
            # override the abstract methods with their parameterised versions
            self.log_likelihood = self.parameterised_log_likelihood
            self.derivatives = self.parameterised_derivatives

        else:
            self.sigma = sigma
            self.inv_sigma = 1.0 / self.sigma
            self.inv_sigma_sqr = self.inv_sigma**2
            self.normalisation = (
                -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_data
            )
            self.parameters = Parameters()
            self.empty_derivatives = {}

    def parameterised_log_likelihood(
        self, predictions: ndarray, **parameters: ndarray
    ) -> float:
        sigma = self.uncertainty_model.get_uncertainties(parameters)
        z = (self.y - predictions) / sigma

        return -0.5 * (z**2).sum() + self.normalisation - log(sigma).sum()

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        sigma, jacobians = self.uncertainty_model.get_uncertainties_and_jacobians(
            parameters
        )
        z = (self.y - predictions) / sigma

        dL_ds = (z**2 - 1) / sigma
        parameter_derivatives = {param: jac @ dL_ds for param, jac in jacobians.items()}
        prediction_derivative = z / sigma
        return prediction_derivative, parameter_derivatives

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        z = (self.y - predictions) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return (self.y - predictions) * self.inv_sigma_sqr, self.empty_derivatives


class LogisticLikelihood(LikelihoodFunction):
    """
    A class for constructing a Logistic likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(self, y_data: ndarray, sigma: ndarray | UncertaintyModel):
        self.y = y_data

        validate_likelihood_data(
            values=y_data, uncertainties=sigma, likelihood_name=self.__class__.__name__
        )

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size

        if isinstance(sigma, UncertaintyModel):
            self.scale_fac = sqrt(3) / pi
            self.uncertainty_model = sigma
            self.parameters = self.uncertainty_model.parameters
            # override the abstract methods with their parameterised versions
            self.log_likelihood = self.parameterised_log_likelihood
            self.derivatives = self.parameterised_derivatives

        else:
            self.sigma = sigma
            self.scale = self.sigma * (sqrt(3) / pi)
            self.inv_scale = 1.0 / self.scale
            self.normalisation = -log(self.scale).sum()
            self.parameters = Parameters()
            self.empty_derivatives = {}

    def parameterised_log_likelihood(
        self, predictions: ndarray, **parameters: ndarray
    ) -> float:
        sigma = self.uncertainty_model.get_uncertainties(parameters)
        scale = sigma * self.scale_fac
        z = (self.y - predictions) / scale

        return z.sum() - 2 * logaddexp(0.0, z).sum() - log(scale).sum()

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        sigma, jacobians = self.uncertainty_model.get_uncertainties_and_jacobians(
            parameters
        )
        scale = sigma * self.scale_fac
        inv_scale = 1 / scale
        z = (self.y - predictions) / scale

        prediction_derivative = (2 / (1 + exp(-z)) - 1) * inv_scale
        dL_ds = (prediction_derivative * z - inv_scale) * self.scale_fac
        parameter_derivatives = {param: jac @ dL_ds for param, jac in jacobians.items()}
        return prediction_derivative, parameter_derivatives

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        z = (self.y - predictions) * self.inv_scale
        return z.sum() - 2 * logaddexp(0.0, z).sum() + self.normalisation

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        z = (self.y - predictions) * self.inv_scale
        return (2 / (1 + exp(-z)) - 1) * self.inv_scale, self.empty_derivatives


class CauchyLikelihood(LikelihoodFunction):
    """
    A class for constructing a Cauchy likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param gamma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(self, y_data: ndarray, gamma: ndarray | UncertaintyModel):
        self.y = y_data
        self.parameters = Parameters()
        self.empty_derivatives = {}

        validate_likelihood_data(
            values=y_data, uncertainties=gamma, likelihood_name=self.__class__.__name__
        )

        # pre-calculate some quantities as an optimisation
        self.n_data = self.y.size

        if isinstance(gamma, UncertaintyModel):
            self.uncertainty_model = gamma
            self.normalisation = -log(pi) * self.n_data
            self.parameters = self.uncertainty_model.parameters
            # override the abstract methods with their parameterised versions
            self.log_likelihood = self.parameterised_log_likelihood
            self.derivatives = self.parameterised_derivatives

        else:
            self.gamma = gamma
            self.inv_gamma = 1.0 / self.gamma
            self.normalisation = -log(pi * self.gamma).sum()
            self.parameters = Parameters()
            self.empty_derivatives = {}

    def parameterised_log_likelihood(
        self, predictions: ndarray, **parameters: ndarray
    ) -> float:
        gamma = self.uncertainty_model.get_uncertainties(parameters)
        z = (self.y - predictions) / gamma

        return -log(1 + z**2).sum() + self.normalisation - log(gamma).sum()

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        gamma, jacobians = self.uncertainty_model.get_uncertainties_and_jacobians(
            parameters
        )
        inv_gamma = 1 / gamma
        z = (self.y - predictions) * inv_gamma

        prediction_derivative = 2 * z / ((1 + z**2) * gamma)
        dL_dg = prediction_derivative * z - inv_gamma
        parameter_derivatives = {param: jac @ dL_dg for param, jac in jacobians.items()}
        return prediction_derivative, parameter_derivatives

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        z = (self.y - predictions) * self.inv_gamma
        return -log(1 + z**2).sum() + self.normalisation

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        z = (self.y - predictions) * self.inv_gamma
        return (2 * self.inv_gamma) * z / (1 + z**2), self.empty_derivatives


class ScaledGaussianLikelihood(LikelihoodFunction):
    """
    A class for constructing a parameterized Gaussian likelihood, where the given
    uncertainties are multiplied by an error-scaling parameter before being used
    in the likelihood calculation.

    This approach of parameterizing the uncertainties can be useful in cases where
    the absolute values of the uncertainties are not well known.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in ``y_data`` as a 1D array.

    :param scale_parameter_name: \
        The name which will be given to the parameter which multiplicatively scales the
        uncertainties given in ``sigma``.
    """

    def __init__(self, y_data: ndarray, sigma: ndarray, scale_parameter_name: str):
        self.y = y_data
        self.sigma = sigma
        self.multiplier = scale_parameter_name
        self.parameters = Parameters((self.multiplier, 1))

        validate_likelihood_data(
            values=y_data, uncertainties=sigma, likelihood_name=self.__class__.__name__
        )

        self.n_data = self.y.size
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_data

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        multiplier = parameters[self.multiplier][0]
        z = (self.y - predictions) * (self.inv_sigma / multiplier)
        return -0.5 * (z**2).sum() + self.normalisation - log(multiplier) * self.n_data

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        multiplier = parameters[self.multiplier]

        z = (self.y - predictions) * (self.inv_sigma / multiplier)
        dL_dp = z * (self.inv_sigma / multiplier)
        dL_dk = ((z**2).sum() - self.n_data) / multiplier
        return dL_dp, {self.multiplier: dL_dk}


def validate_likelihood_data(
    values: ndarray, uncertainties: ndarray | UncertaintyModel, likelihood_name: str
):
    if not isinstance(values, ndarray):
        raise TypeError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The data values must be an instance of numpy.ndarray.
            \r>> Instead, the given type was:
            \r>> {type(values)}
            """
        )

    if not isfinite(values).all():
        raise ValueError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The data values array must contain only finite values.
            """
        )

    if values.ndim != 1:
        raise ValueError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The data values array must have only one dimension, but
            \r>> instead has shape:
            \r>> {values.shape}
            """
        )

    if isinstance(uncertainties, UncertaintyModel):
        good_parameters = (
            hasattr(uncertainties, "parameters")
            and isinstance(uncertainties.parameters, Parameters)
            and len(uncertainties.parameters) > 0
        )
        if not good_parameters:
            raise ValueError(
                f"""\n
                \r[ {likelihood_name} error ]
                \r>> The given UncertaintyModel object must have a 'parameters'
                \r>> attribute which is an instance of the 'Parameters' class, and
                \r>> specifies at least one free parameter.
                """
            )

    elif isinstance(uncertainties, ndarray):
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

        valid_uncertainties = (
            isfinite(uncertainties).all()
            and (uncertainties > 0.0).all()
        )
        if not valid_uncertainties:
            raise ValueError(
                f"""\n
                \r[ {likelihood_name} error ]
                \r>> The uncertainties array must contain only finite
                \r>> values, and all uncertainties must have values greater than zero.
                """
            )
    else:
        raise TypeError(
            f"""\n
            \r[ {likelihood_name} error ]
            \r>> The uncertainties argument must either be an instance of numpy.ndarray
            \r>> or the UncertaintyModel class.
            """
        )
