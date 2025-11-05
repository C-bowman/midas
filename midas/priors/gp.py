from numpy import array, diagonal, eye, log, ndarray
from numpy.linalg import cholesky, LinAlgError
from scipy.linalg import solve_triangular
from warnings import warn
from inference.gp.covariance import CovarianceFunction, SquaredExponential
from inference.gp.mean import MeanFunction, ConstantMean

from midas.parameters import ParameterVector, FieldRequest
from midas.priors.base import BasePrior


class GaussianProcessPrior(BasePrior):
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

        if field_positions is not None:
            self.target = field_positions.name
            spatial_data = array([v for v in field_positions.coordinates.values()]).T
            self.field_requests = [field_positions]
            self.parameters = []
            self.I = eye(field_positions.size)

        elif parameter_coordinates is not None and parameters is not None:
            self.target = parameters.name
            spatial_data = array([v for v in parameter_coordinates.values()]).T
            self.field_requests = []
            self.parameters = [parameters]
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
        self.parameters.extend(
            [
                ParameterVector(name=self.cov_tag, size=self.cov.n_params),
                ParameterVector(name=self.mean_tag, size=self.mean.n_params),
            ]
        )

    def probability(self, **kwargs):
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

    def gradients(self, **kwargs):
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
