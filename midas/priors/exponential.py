from numpy import ndarray, atleast_1d, zeros
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
from midas.state import BasePrior


class ExponentialPrior(BasePrior):
    """
    Specify a exponential prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the exponential prior.

    :param mean: \
        The mean of the exponential prior corresponding to each parameter or requested
        field value.

    :param field_positions: \
        A ``FieldRequest`` specifying the field and coordinates to which the exponential
        prior will be applied. If specified, ``field_positions`` will override
        any values passed to the ``parameters`` arguments.

    :param parameter_vector: \
        A ``ParameterVector`` specifying which parameters to which the exponential prior
        will be applied.
    """

    def __init__(
        self,
        name: str,
        mean: ndarray,
        field_positions: FieldRequest = None,
        parameter_vector: ParameterVector = None,
    ):

        self.name = name
        self.mean = atleast_1d(mean)
        self.lam = 1.0 / self.mean

        if isinstance(field_positions, FieldRequest):
            self.target = field_positions.name
            self.n_targets = field_positions.size
            self.fields = Fields(field_positions)
            self.parameters = Parameters()

        elif isinstance(parameter_vector, ParameterVector):
            self.target = parameter_vector.name
            self.n_targets = parameter_vector.size
            self.fields = Fields()
            self.parameters = Parameters(parameter_vector)

        else:
            raise ValueError(
                """\n
                \r[ ExponentialPrior error ]
                \r>> One of the 'field_positions' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        assert self.mean.ndim == 1
        assert self.mean.size == self.n_targets
        assert isinstance(name, str)

    def probability(self, **kwargs: ndarray) -> float:
        target_values = kwargs[self.target]
        if (target_values < 0.).any():
            return -1e-50
        else:
            z = -self.lam * target_values
            return z.sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        target_values = kwargs[self.target]
        if (target_values < 0.).any():
            return {self.target: zeros(self.n_targets)}
        else:
            return {self.target: -self.lam}
