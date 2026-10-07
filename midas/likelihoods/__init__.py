from numpy import ndarray, log, exp, logaddexp, sqrt, pi, isfinite, where
from midas.posterior import LikelihoodFunction
from midas.parameters import Parameters
from midas.likelihoods.uncertainties import UncertaintyModel
from midas.likelihoods.uncertainties import ConstantUncertainty, LinearUncertainty


class GaussianLikelihood(LikelihoodFunction):
    """
    A class for constructing a Gaussian likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in ``y_data`` as a 1D array.
        Alternatively, a model for the uncertainties (inheriting from the
        ``UncertaintyModel`` base-class) can be provided, allowing the uncertainties
        to be parameterised and inferred.
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
        sigma = self.uncertainty_model.uncertainties(parameters)
        z = (self.y - predictions) / sigma

        return -0.5 * (z**2).sum() + self.normalisation - log(sigma).sum()

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        sigma, pullback = self.uncertainty_model.uncertainties_and_pullback(
            parameters
        )
        z = (self.y - predictions) / sigma

        dL_ds = (z**2 - 1) / sigma
        parameter_derivatives = pullback(dL_ds)
        prediction_derivative = z / sigma
        return prediction_derivative, parameter_derivatives

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        z = (self.y - predictions) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return (self.y - predictions) * self.inv_sigma_sqr, self.empty_derivatives


class SplitGaussianLikelihood(LikelihoodFunction):
    """
    A class for constructing a split Gaussian likelihood function.

    The distribution uses ``sigma_lower`` when a measured value is below its
    prediction, and ``sigma_upper`` when it is above its prediction. The two
    uncertainties must either both be arrays or both be uncertainty models.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma_lower: \
        The standard deviations for downward fluctuations in ``y_data`` as a 1D array.
        Alternatively, a model for these uncertainties (inheriting from the
        ``UncertaintyModel`` base-class) can be provided.

    :param sigma_upper: \
        The standard deviations for upward fluctuations in ``y_data`` as a 1D array.
        Alternatively, a model for these uncertainties (inheriting from the
        ``UncertaintyModel`` base-class) can be provided.
    """

    def __init__(
        self,
        y_data: ndarray,
        sigma_lower: ndarray | UncertaintyModel,
        sigma_upper: ndarray | UncertaintyModel,
    ):
        self.y = y_data

        validate_likelihood_data(
            values=y_data,
            uncertainties=sigma_lower,
            likelihood_name=self.__class__.__name__,
        )
        validate_likelihood_data(
            values=y_data,
            uncertainties=sigma_upper,
            likelihood_name=self.__class__.__name__,
        )

        self.n_data = self.y.size
        lower_is_model = isinstance(sigma_lower, UncertaintyModel)
        upper_is_model = isinstance(sigma_upper, UncertaintyModel)
        if lower_is_model != upper_is_model:
            raise ValueError(
                f"""\n
                \r[ {self.__class__.__name__} error ]
                \r>> The lower and upper uncertainties must either both be arrays
                \r>> or both be instances of the UncertaintyModel class.
                """
            )

        if lower_is_model and upper_is_model:
            self.lower_uncertainty_model = sigma_lower
            self.upper_uncertainty_model = sigma_upper
            parameters_by_name = {}
            for model in (
                self.lower_uncertainty_model,
                self.upper_uncertainty_model,
            ):
                for parameter in model.parameters:
                    existing = parameters_by_name.get(parameter.name)
                    if existing is not None and existing.size != parameter.size:
                        raise ValueError(
                            f"""\n
                            \r[ {self.__class__.__name__} error ]
                            \r>> The lower and upper uncertainty models contain
                            \r>> parameters which share the name '{parameter.name}'
                            \r>> but have different sizes.
                            """
                        )
                    parameters_by_name[parameter.name] = parameter

            self.parameters = Parameters(*parameters_by_name.values())
            self.normalisation = 0.5 * log(2 / pi) * self.n_data
            self.log_likelihood = self.parameterised_log_likelihood
            self.derivatives = self.parameterised_derivatives

        else:
            self.sigma_lower = sigma_lower
            self.sigma_upper = sigma_upper
            sigma_sum = self.sigma_lower + self.sigma_upper
            self.inv_sigma_lower_sqr = 1.0 / self.sigma_lower**2
            self.inv_sigma_upper_sqr = 1.0 / self.sigma_upper**2
            self.normalisation = (
                0.5 * log(2 / pi) * self.n_data - log(sigma_sum).sum()
            )
            self.parameters = Parameters()
            self.empty_derivatives = {}

    def _uncertainties(
        self, parameters: dict[str, ndarray]
    ) -> tuple[ndarray, ndarray]:
        return (
            self.lower_uncertainty_model.uncertainties(parameters),
            self.upper_uncertainty_model.uncertainties(parameters),
        )

    def parameterised_log_likelihood(
        self, predictions: ndarray, **parameters: ndarray
    ) -> float:
        sigma_lower, sigma_upper = self._uncertainties(parameters)
        residual = self.y - predictions
        sigma = where(residual < 0, sigma_lower, sigma_upper)
        z = residual / sigma

        return (
            -0.5 * (z**2).sum()
            + self.normalisation
            - log(sigma_lower + sigma_upper).sum()
        )

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        sigma_lower, lower_pullback = (
            self.lower_uncertainty_model.uncertainties_and_pullback(parameters)
        )
        sigma_upper, upper_pullback = (
            self.upper_uncertainty_model.uncertainties_and_pullback(parameters)
        )

        residual = self.y - predictions
        lower_side = residual < 0
        sigma = where(lower_side, sigma_lower, sigma_upper)
        inv_sigma_sum = 1 / (sigma_lower + sigma_upper)

        prediction_derivative = residual / sigma**2
        dL_ds_lower = (
            where(lower_side, residual**2 / sigma_lower**3, 0.0)
            - inv_sigma_sum
        )
        dL_ds_upper = (
            where(lower_side, 0.0, residual**2 / sigma_upper**3)
            - inv_sigma_sum
        )
        parameter_derivatives = lower_pullback(dL_ds_lower)
        for name, derivative in upper_pullback(dL_ds_upper).items():
            if name in parameter_derivatives:
                parameter_derivatives[name] += derivative
            else:
                parameter_derivatives[name] = derivative

        return prediction_derivative, parameter_derivatives

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        residual = self.y - predictions
        inv_sigma_sqr = where(
            residual < 0, self.inv_sigma_lower_sqr, self.inv_sigma_upper_sqr
        )
        return -0.5 * (residual**2 * inv_sigma_sqr).sum() + self.normalisation

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        residual = self.y - predictions
        inv_sigma_sqr = where(
            residual < 0, self.inv_sigma_lower_sqr, self.inv_sigma_upper_sqr
        )
        return residual * inv_sigma_sqr, self.empty_derivatives


class LogisticLikelihood(LikelihoodFunction):
    """
    A class for constructing a Logistic likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The uncertainties corresponding to each element in ``y_data`` as a 1D array.
        Alternatively, a model for the uncertainties (inheriting from the
        ``UncertaintyModel`` base-class) can be provided, allowing the uncertainties
        to be parameterised and inferred.
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
        sigma = self.uncertainty_model.uncertainties(parameters)
        scale = sigma * self.scale_fac
        z = (self.y - predictions) / scale

        return z.sum() - 2 * logaddexp(0.0, z).sum() - log(scale).sum()

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        sigma, pullback = self.uncertainty_model.uncertainties_and_pullback(
            parameters
        )
        scale = sigma * self.scale_fac
        inv_scale = 1 / scale
        z = (self.y - predictions) / scale

        prediction_derivative = (2 / (1 + exp(-z)) - 1) * inv_scale
        dL_ds = (prediction_derivative * z - inv_scale) * self.scale_fac
        parameter_derivatives = pullback(dL_ds)
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
        Alternatively, a model for the uncertainties (inheriting from the
        ``UncertaintyModel`` base-class) can be provided, allowing the uncertainties
        to be parameterised and inferred.
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
        gamma = self.uncertainty_model.uncertainties(parameters)
        z = (self.y - predictions) / gamma

        return -log(1 + z**2).sum() + self.normalisation - log(gamma).sum()

    def parameterised_derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        gamma, pullback = self.uncertainty_model.uncertainties_and_pullback(
            parameters
        )
        inv_gamma = 1 / gamma
        z = (self.y - predictions) * inv_gamma

        prediction_derivative = 2 * z / ((1 + z**2) * gamma)
        dL_dg = prediction_derivative * z - inv_gamma
        parameter_derivatives = pullback(dL_dg)
        return prediction_derivative, parameter_derivatives

    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        z = (self.y - predictions) * self.inv_gamma
        return -log(1 + z**2).sum() + self.normalisation

    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        z = (self.y - predictions) * self.inv_gamma
        return (2 * self.inv_gamma) * z / (1 + z**2), self.empty_derivatives


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
            isfinite(uncertainties).all() and (uncertainties > 0.0).all()
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
