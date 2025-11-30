from numpy import ndarray, atleast_1d
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, FieldRequests
from midas.state import BasePrior


class GaussianPrior(BasePrior):
    """
    Specify a Gaussian prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the Gaussian prior

    :param mean: \
        The mean of the Gaussian prior corresponding to each parameter or requested
        field value.

    :param standard_deviation: \
        The standard deviation of the Gaussian prior corresponding to each parameter
        or requested field value.

    :param field_positions: \
        A ``FieldRequest`` specifying the field and coordinates to which the Gaussian
        prior will be applied. If specified, ``field_positions`` will override
        any values passed to the ``parameters`` arguments.

    :param parameter_vector: \
        A ``ParameterVector`` specifying which parameters to which the Gaussian prior
        will be applied.
    """

    def __init__(
        self,
        name: str,
        mean: ndarray,
        standard_deviation: ndarray,
        field_positions: FieldRequest = None,
        parameter_vector: ParameterVector = None,
    ):

        self.name = name
        self.mean = atleast_1d(mean)
        self.sigma = atleast_1d(standard_deviation)
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2

        if isinstance(field_positions, FieldRequest):
            self.target = field_positions.name
            self.n_targets = field_positions.size
            self.field_requests = FieldRequests(field_positions)
            self.parameters = Parameters()

        elif isinstance(parameter_vector, ParameterVector):
            self.target = parameter_vector.name
            self.n_targets = parameter_vector.size
            self.field_requests = FieldRequests()
            self.parameters = Parameters(parameter_vector)

        else:
            raise ValueError(
                """\n
                \r[ GaussianPrior error ]
                \r>> One of the 'field_positions' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        assert self.mean.ndim == self.sigma.ndim == 1
        assert self.mean.size == self.sigma.size == self.n_targets
        assert isinstance(name, str)

    def probability(self, **kwargs: ndarray) -> float:
        target_values = kwargs[self.target]
        z = (target_values - self.mean) * self.inv_sigma
        return -0.5 * (z**2).sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        target_values = kwargs[self.target]
        return {self.target: -(target_values - self.mean) * self.inv_sigma_sqr}
