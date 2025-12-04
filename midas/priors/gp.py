from numpy import array, diagonal, eye, log, ndarray
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import solve_triangular
from warnings import warn
from inference.gp.covariance import CovarianceFunction, SquaredExponential
from inference.gp.mean import MeanFunction, ConstantMean

from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
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
        construct the GP prior. If specified, ``field_positions`` will override
        any values passed to the ``parameters`` or ``parameter_coordinates`` arguments.

    :param parameters: \
        A ``ParameterVector`` specifying which parameters will be used as inputs
        to the GP prior.

    :param parameter_coordinates: \
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
        parameter_coordinates: dict[str, ndarray] = None,
    ):
        self.cov = covariance
        self.mean = mean
        self.name = name

        if isinstance(field_positions, FieldRequest):
            self.target = field_positions.name
            spatial_data = array([v for v in field_positions.coordinates.values()]).T
            self.fields = Fields(field_positions)
            target_parameters = []
            self.I = eye(field_positions.size)

        elif isinstance(parameters, ParameterVector) and isinstance(parameter_coordinates, dict):
            self.target = parameters.name
            spatial_data = array([v for v in parameter_coordinates.values()]).T
            self.fields = Fields()
            target_parameters = [parameters]
            self.I = eye(parameters.size)

        else:
            raise ValueError(
                """\n
                \r[ GaussianProcessPrior error ]
                \r>> Either the 'field_positions' argument, or both of the 'parameters'
                \r>> and 'parameter_coordinates' arguments must be provided.
                """
            )

        self.cov.pass_spatial_data(spatial_data)
        self.mean.pass_spatial_data(spatial_data)

        self.cov_tag = f"{self.name}_cov_hyperpars"
        self.mean_tag = f"{self.name}_mean_hyperpars"
        self.hyperparameters = {
            self.cov_tag: self.cov.hyperpar_labels,
            self.mean_tag: self.mean.hyperpar_labels
        }

        self.parameters = Parameters(
            ParameterVector(name=self.cov_tag, size=self.cov.n_params),
            ParameterVector(name=self.mean_tag, size=self.mean.n_params),
            *target_parameters
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
