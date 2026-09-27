from numpy import full, ndarray, atleast_1d, log, zeros
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
from midas.posterior import BasePrior
from midas.validation import validate_numeric_input, validate_name


class BetaPrior(BasePrior):
    """
    Specify a beta distribution prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the beta prior.

    :param alpha: \
        The ``alpha`` shape parameter of the beta prior. This may be a positive
        ``float``, in which case the same value is used for every parameter or
        requested field value, or a one-dimensional ``numpy.ndarray`` containing
        one positive value for each target.

    :param beta: \
        The ``beta`` shape parameter of the beta prior. This may be a positive
        ``float``, in which case the same value is used for every parameter or
        requested field value, or a one-dimensional ``numpy.ndarray`` containing
        one positive value for each target.

    :param field_request: \
        A ``FieldRequest`` specifying the field and coordinates to which the beta
        prior will be applied. If specified, ``field_request`` will override
        any values passed to the ``parameter_vector`` arguments.

    :param parameter_vector: \
        A ``ParameterVector`` specifying the parameters to which the beta prior
        will be applied.

    :param limits: \
        A tuple of two floats specifying the range of values to which the prior is
        applied. The Beta distribution normally only supports values between 0 and 1,
        so the limits are used to re-scale values in the given range to [0, 1].
    """

    def __init__(
        self,
        name: str,
        alpha: ndarray | float,
        beta: ndarray | float,
        field_request: FieldRequest | None = None,
        parameter_vector: ParameterVector | None = None,
        limits: tuple[float, float] = (0, 1),
    ):

        validate_name(name, error_source="BetaPrior")
        self.name = name

        assert hasattr(limits, "__len__") and len(limits) == 2
        assert limits[0] < limits[1]
        lwr, upr = limits

        self.scale = 1 / (upr - lwr)
        self.offset = -lwr * self.scale

        if isinstance(field_request, FieldRequest):
            self.target = field_request.name
            self.target_type = "field_request"
            self.n_targets = field_request.size
            self.fields = Fields(field_request)
            self.parameters = Parameters()

        elif isinstance(parameter_vector, ParameterVector):
            self.target = parameter_vector.name
            self.target_type = "parameter_vector"
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

        self.alpha = atleast_1d(alpha)
        self.alpha = full(self.n_targets, self.alpha) if self.alpha.size == 1 else self.alpha
        validate_numeric_input(
            values=self.alpha,
            shape=(self.n_targets,),
            shape_name=self.target_type,
            error_source="BetaPrior",
            input_name="alpha",
            limits=(0.0, float("inf")),
            strict_limits=True,
        )

        self.beta = atleast_1d(beta)
        self.beta = full(self.n_targets, self.beta) if self.beta.size == 1 else self.beta
        validate_numeric_input(
            values=self.beta,
            shape=(self.n_targets,),
            shape_name=self.target_type,
            error_source="BetaPrior",
            input_name="beta",
            limits=(0.0, float("inf")),
            strict_limits=True,
        )

        self.am1 = self.alpha - 1
        self.bm1 = self.beta - 1

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
