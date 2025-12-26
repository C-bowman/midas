from numpy import ndarray, atleast_1d, log, full
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
from midas.state import BasePrior


class BetaPrior(BasePrior):
    """
    Specify a exponential prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the exponential prior.

    :param alpha: \
        The 'alpha' shape parameter of the beta prior corresponding to each parameter or
        requested field value. All values of 'alpha' must be greater than zero.

    :param beta: \
        The 'beta' shape parameter of the beta prior corresponding to each parameter or
        requested field value. All values of 'beta' must be greater than zero.

    :param field_positions: \
        A ``FieldRequest`` specifying the field and coordinates to which the exponential
        prior will be applied. If specified, ``field_positions`` will override
        any values passed to the ``parameters`` arguments.

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
        field_positions: FieldRequest = None,
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
                \r[ BetaPrior error ]
                \r>> One of the 'field_positions' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        assert self.alpha.ndim == self.beta.ndim == 1
        assert self.alpha.size == self.beta.size == self.n_targets
        assert isinstance(name, str)

    def probability(self, **kwargs: ndarray) -> float:
        target_values = kwargs[self.target]
        z = self.scale * target_values + self.offset
        log_prob = full(self.n_targets, fill_value=-1e50)
        valid = (z > 0.) & (z < 1.)
        if valid.any():
            inds = valid.nonzero()
            log_prob[inds] = self.am1[inds] * log(z[inds]) + self.bm1[inds] * log(1 - z[inds])
        return log_prob.sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        target_values = kwargs[self.target]
        z = self.scale * target_values + self.offset
        gradient = full(self.n_targets, fill_value=-1e50)
        valid = (z > 0.) & (z < 1.)
        if valid.any():
            inds = valid.nonzero()
            gradient[inds] = (self.am1[inds] / z - self.bm1[inds] / (1 - z[inds])) * self.scale
        return {self.target: gradient}
