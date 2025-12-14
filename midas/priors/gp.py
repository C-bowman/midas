from numpy import array, diagonal, eye, log, ndarray
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import solve_triangular
from warnings import warn
from inference.gp.covariance import CovarianceFunction, SquaredExponential
from inference.gp.mean import MeanFunction, ConstantMean

from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields, validate_coordinates
from midas.state import BasePrior


class GaussianProcessPrior(BasePrior):
    """
    Specify a Gaussian process prior over either a series of field values, or a
    set of parameters and corresponding spatial coordinates.

    :param name: \
        The name used to identify the GP prior.

    :param covariance: \
        An instance of a ``CovarianceFunction`` class from the ``inference-tools`` package.

    :param mean: \
        An instance of a ``MeanFunction`` class from the ``inference-tools`` package.

    :param field_positions: \
        A ``FieldRequest`` specifying the field and coordinates which will be used to
        construct the GP prior. By default, the coordinates in the given ``FieldRequest``
        will be used as the spatial coordinates for the GP, however these can be
        overridden by specifying the ``coordinates`` keyword argument. This enables
        the GP to use a different coordinate system from the one in which the field is
        being modelled if desired.

        If specified, ``field_positions`` will override any values passed to
        the ``parameters`` argument.

    :param parameters: \
        A ``ParameterVector`` specifying which parameters will be used as inputs
        to the GP prior.

    :param coordinates: \
        A set of coordinates (a dictionary mapping coordinate names as ``str`` to
        coordinate values as ``numpy.ndarray``) corresponding the ``ParameterVector``
        passed to the ``parameters`` argument.
    """

    def __init__(
        self,
        name: str,
        covariance: CovarianceFunction = SquaredExponential(),
        mean: MeanFunction = ConstantMean(),
        field_positions: FieldRequest = None,
        parameters: ParameterVector = None,
        coordinates: dict[str, ndarray] = None,
    ):
        self.cov = covariance
        self.mean = mean
        self.name = name

        if coordinates is not None:
            validate_coordinates(coordinates, error_source="GaussianProcessPrior")

        if isinstance(field_positions, FieldRequest):
            self.target = field_positions.name
            if coordinates is not None:
                assert all(field_positions.size == c.size for c in coordinates.values())
                spatial_data = array([v for v in coordinates.values()]).T
            else:
                spatial_data = array(
                    [v for v in field_positions.coordinates.values()]
                ).T
            self.fields = Fields(field_positions)
            target_parameters = []
            self.I = eye(field_positions.size)

        elif isinstance(parameters, ParameterVector) and isinstance(coordinates, dict):
            self.target = parameters.name
            spatial_data = array([v for v in coordinates.values()]).T
            self.fields = Fields()
            target_parameters = [parameters]
            self.I = eye(parameters.size)

        else:
            raise ValueError(
                """\n
                \r[ GaussianProcessPrior error ]
                \r>> Either the 'field_positions' argument, or both of the 'parameters'
                \r>> and 'coordinates' arguments must be provided.
                """
            )

        self.cov.pass_spatial_data(spatial_data)
        self.mean.pass_spatial_data(spatial_data)

        self.cov_tag = f"{self.name}_cov_hyperpars"
        self.mean_tag = f"{self.name}_mean_hyperpars"
        self.hyperparameters = {
            self.cov_tag: self.cov.hyperpar_labels,
            self.mean_tag: self.mean.hyperpar_labels,
        }

        self.parameters = Parameters(
            (self.cov_tag, self.cov.n_params),
            (self.mean_tag, self.mean.n_params),
            *target_parameters,
        )

    def probability(self, **kwargs: ndarray) -> float:
        field_values = kwargs[self.target]
        K = self.cov.build_covariance(kwargs[self.cov_tag])
        mu = self.mean.build_mean(kwargs[self.mean_tag])

        try:  # protection against singular matrix error crash
            L = cholesky(K)
            v = solve_triangular(L, field_values - mu, lower=True)
            return -0.5 * (v @ v) - log(diagonal(L)).sum()
        except LinAlgError:
            warn("Cholesky decomposition failure in marginal_likelihood")
            return -1e50

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        K, grad_K = self.cov.covariance_and_gradients(kwargs[self.cov_tag])
        mu, grad_mu = self.mean.mean_and_gradients(kwargs[self.mean_tag])

        # Use the cholesky decomposition to get the inverse-covariance
        L = cholesky(K)
        iK = solve_triangular(L, self.I, lower=True)
        iK = iK.T @ iK

        # calculate some quantities we need for the derivatives
        dy = kwargs[self.target] - mu
        alpha = iK @ dy
        Q = alpha[:, None] * alpha[None, :] - iK

        return {
            self.target: -alpha,
            self.mean_tag: array([(alpha * dmu).sum() for dmu in grad_mu]),
            self.cov_tag: array([0.5 * (Q * dK.T).sum() for dK in grad_K]),
        }

    def fix_hyperparameters(self, hyperparameters: dict[str, ndarray]):
        """
        Fix the value of the mean and covariance hyperparameters. As this alters the
        overall set of parameters in the analysis, ``fix_hyperparameters`` should be
        called before building the posterior distribution by calling
        ``PlasmaState.build_posterior``.

        :param hyperparameters: \
            A dictionary mapping the names of each of the mean and covariance
            hyperparameters to their corresponding values.
        """
        cov_params = hyperparameters[self.cov_tag]
        mean_params = hyperparameters[self.mean_tag]

        # filter the parameters down to just the target parameters
        self.parameters = Parameters(
            *[p for p in self.parameters if p.name == self.target]
        )

        # calculate the fixed mean and covariance
        self.K = self.cov.build_covariance(cov_params)
        self.mu = self.mean.build_mean(mean_params)
        L = cholesky(self.K)
        iK = solve_triangular(L, self.I, lower=True)
        self.iK = iK.T @ iK
        self.logdet = log(diagonal(L)).sum()

        # override required abstract methods with fixed-hyperparameter variants
        self.probability = self.__fixed_probability
        self.gradients = self.__fixed_gradients

    def __fixed_probability(self, **kwargs: ndarray) -> float:
        dy = kwargs[self.target] - self.mu
        z = dy @ (self.iK @ dy)
        return -0.5 * z - self.logdet

    def __fixed_gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        # calculate some quantities we need for the derivatives
        dy = kwargs[self.target] - self.mu
        alpha = self.iK @ dy
        return {self.target: -alpha}
