from numpy import eye, ndarray, maximum, atleast_1d
from midas.state import BasePrior
from midas.parameters import Parameters, Fields, FieldRequest, ParameterVector
from midas.validation import validate_numeric_input, validate_name


class SoftLimitPrior(BasePrior):
    """
    A prior which is uniform up to a set of upper limits, and beyond which is
    Gaussian with a corresponding set of standard deviations. This allows
    element-wise 'soft' limits to be imposed on field values or parameters.

    Alternatively, a 2D matrix operator can also be given, and the prior will instead
    be applied to the result of the matrix multiplication of that operator and the
    vector of target values. The ``upper_limit`` and ``standard_deviation`` arrays
    must then have one element for each row of the operator.

    :param name: \
        The name used to identify the prior.

    :param upper_limit: \
        A one-dimensional array containing the upper limit beyond which a Gaussian
        penalty is applied to each target value. Without an operator, its shape must
        match the target vector. With an operator, its shape must be
        ``(operator.shape[0],)``.

    :param standard_deviation: \
        A one-dimensional array containing the positive standard deviation of the
        Gaussian penalty for each upper limit. It must have the same shape as
        ``upper_limit``.

    :param field_request: \
        A ``FieldRequest`` specifying the field values to which the soft-limit prior is applied.

    :param parameter_vector: \
        A ``ParameterVector`` specifying the parameters to which the soft-limit prior is applied.

    :param operator: \
        A linear operator (as a 2D array) which matrix-multiplies the vector of
        target values. If specified, the prior will be applied to the result of this
        matrix multiplication instead of directly to the target values.
    """

    def __init__(
        self,
        name: str,
        upper_limit: ndarray,
        standard_deviation: ndarray,
        field_request: FieldRequest | None = None,
        parameter_vector: ParameterVector | None = None,
        operator: ndarray | None = None,
    ):
        validate_name(name, error_source="SoftLimitPrior")
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
                \r[ SoftLimitPrior error ]
                \r>> One of the 'field_request' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        self.A = operator if operator is not None else eye(self.n_targets)
        assert isinstance(self.A, ndarray)
        assert self.A.ndim == 2
        assert self.A.shape[1] == self.n_targets
        output_shape = (self.A.shape[0],)

        self.limit = atleast_1d(upper_limit)
        validate_numeric_input(
            values=self.limit,
            shape=output_shape,
            shape_name="operator output",
            error_source="SoftLimitPrior",
            input_name="upper_limit",
        )

        self.sigma = atleast_1d(standard_deviation)
        validate_numeric_input(
            values=self.sigma,
            shape=output_shape,
            shape_name="operator output",
            error_source="SoftLimitPrior",
            input_name="standard_deviation",
            limits=(0.0, float("inf")),
            strict_limits=True,
        )

        self.weight = 1.0 / self.sigma**2

    def probability(self, **kwargs: ndarray) -> float:
        v = kwargs[self.target]
        z = self.A @ v - self.limit
        return -0.5 * (self.weight * maximum(z, 0.0) ** 2).sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        v = kwargs[self.target]
        z = self.A @ v - self.limit
        return {
            self.target: -self.A.T @ (self.weight * maximum(z, 0.0))
        }
