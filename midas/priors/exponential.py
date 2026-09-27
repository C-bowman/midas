from numpy import full, ndarray, atleast_1d, zeros
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
from midas.posterior import BasePrior
from midas.validation import validate_numeric_input, validate_name


class ExponentialPrior(BasePrior):
    """
    Specify an exponential prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the exponential prior.

    :param mean: \
        The mean of the exponential prior. This may be a positive ``float``, in
        which case the same value is used for every target, or a one-dimensional
        ``numpy.ndarray`` containing one positive value for each target.

    :param field_request: \
        A ``FieldRequest`` specifying the field and coordinates to which the exponential
        prior will be applied. If specified, ``field_request`` will override
        any values passed to the ``parameter_vector`` arguments.

    :param parameter_vector: \
        A ``ParameterVector`` specifying the parameters to which the exponential prior
        will be applied.
    """

    def __init__(
        self,
        name: str,
        mean: ndarray | float,
        field_request: FieldRequest | None = None,
        parameter_vector: ParameterVector | None = None,
    ):
        validate_name(name, error_source="ExponentialPrior")
        self.name = name

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
                \r[ ExponentialPrior error ]
                \r>> One of the 'field_request' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        self.mean = atleast_1d(mean)
        self.mean = full(self.n_targets, self.mean) if self.mean.size == 1 else self.mean
        validate_numeric_input(
            values=self.mean,
            shape=(self.n_targets,),
            shape_name=self.target_type,
            error_source="ExponentialPrior",
            input_name="mean",
            limits=(0.0, float("inf")),
            strict_limits=True,
        )

        self.lam = 1.0 / self.mean

    def probability(self, **kwargs: ndarray) -> float:
        target_values = kwargs[self.target]
        if (target_values < 0.).any():
            return -1e50
        else:
            z = -self.lam * target_values
            return z.sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        target_values = kwargs[self.target]
        if (target_values < 0.).any():
            return {self.target: zeros(self.n_targets)}
        else:
            return {self.target: -self.lam}
