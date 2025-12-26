from numpy import ndarray, atleast_1d, log, zeros
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
from midas.state import BasePrior


class BetaPrior(BasePrior):
    """
    Specify a beta distribution prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the exponential prior.

    :param alpha: \
        The 'alpha' shape parameter of the beta prior corresponding to each parameter or
        requested field value. All values of 'alpha' must be greater than zero.

    :param beta: \
        The 'beta' shape parameter of the beta prior corresponding to each parameter or
        requested field value. All values of 'beta' must be greater than zero.

    :param field_request: \
        A ``FieldRequest`` specifying the field and coordinates to which the exponential
        prior will be applied. If specified, ``field_request`` will override
        any values passed to the ``parameter_vector`` arguments.

    :param parameter_vector: \
        A ``ParameterVector`` specifying which parameters to which the exponential prior
        will be applied.

    :param limits: \
        A tuple of two floats specifying the range of values to which the prior is
        applied. The Beta distribution normally only supports values between 0 and 1,
        so the limits are used to re-scale values in the given range to [0, 1].
    """

    def __init__(
        self,
        name: str,
        alpha: ndarray,
        beta: ndarray,
        field_request: FieldRequest = None,
        parameter_vector: ParameterVector = None,
        limits: tuple[float, float] = (0, 1),
    ):

        self.name = name
        self.alpha = atleast_1d(alpha)
        self.beta = atleast_1d(beta)

        self.am1 = self.alpha - 1
        self.bm1 = self.beta - 1

        lwr, upr = limits
        assert hasattr(limits, "__len__") and len(limits) == 2
        assert lwr < upr
        self.scale = 1 / (upr - lwr)
        self.offset = -lwr * self.scale

        if isinstance(field_request, FieldRequest):
            self.target = field_request.name
            self.n_targets = field_request.size
            self.fields = Fields(field_request)
            self.parameters = Parameters()

        elif isinstance(parameter_vector, ParameterVector):
            self.target = parameter_vector.name
            self.n_targets = parameter_vector.size
            self.fields = Fields()
            self.parameters = Parameters(parameter_vector)

        else:
            raise ValueError(
                """\n
                \r[ BetaPrior error ]
                \r>> One of the 'field_request' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        assert self.alpha.ndim == self.beta.ndim == 1
        assert self.alpha.size == self.beta.size == self.n_targets
        assert (self.alpha > 0).all() and (self.beta > 0).all()
        assert isinstance(name, str)

    def probability(self, **kwargs: ndarray) -> float:
        target_values = kwargs[self.target]
        z = self.scale * target_values + self.offset
        invalid = (z <= 0.) | (z >= 1.)
        if invalid.any():
            return -1e50
        else:
            log_prob = self.am1 * log(z) + self.bm1 * log(1 - z)
            return log_prob.sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        target_values = kwargs[self.target]
        z = self.scale * target_values + self.offset

        invalid = (z <= 0.) | (z >= 1.)
        if invalid.any():
            return {self.target: zeros(self.n_targets)}
        else:
            gradient = (self.am1 / z - self.bm1 / (1 - z)) * self.scale
            return {self.target: gradient}
