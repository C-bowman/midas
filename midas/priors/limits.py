from numpy import eye, full, ndarray, maximum, atleast_1d
from scipy.sparse import sparray
from midas.posterior import BasePrior
from midas.parameters import Parameters, Fields, FieldRequest, ParameterVector
from midas.validation import validate_numeric_input, validate_name


class SoftLimitPrior(BasePrior):
    """
    A prior which is uniform up to a set of upper limits, and beyond which is
    Gaussian with a corresponding set of standard deviations. This allows
    element-wise 'soft' limits to be imposed on field values or parameters.

    Alternatively, a 2D matrix operator can also be given, and the prior will instead
    be applied to the result of the matrix multiplication of that operator and the
    vector of target values. Array inputs for ``upper_limit`` and
    ``standard_deviation`` must then have one element for each row of the operator.

    :param name: \
        The name used to identify the prior.

    :param upper_limit: \
        The upper limit beyond which a Gaussian penalty is applied. This may be a
        ``float``, in which case the same limit is used for every operator output,
        or a one-dimensional ``numpy.ndarray``. Without an operator, the array shape
        must match the target vector. With an operator, it must be
        ``(operator.shape[0],)``.

    :param standard_deviation: \
        The standard deviation of the Gaussian penalty. This may be a positive
        ``float``, in which case the same value is used for every operator output,
        or a one-dimensional ``numpy.ndarray`` containing one positive value per
        operator output.

    :param field_request: \
        A ``FieldRequest`` specifying the field values to which the soft-limit prior is applied.

    :param parameter_vector: \
        A ``ParameterVector`` specifying the parameters to which the soft-limit prior is applied.

    :param operator: \
        A linear operator (as a 2D ``numpy.ndarray`` or ``scipy.sparse`` array)
        which matrix-multiplies the vector of target values. If specified,
        the prior will be applied to the result of this matrix multiplication
        instead of directly to the target values. Sparse operators are stored
        without densifying them.
    """

    def __init__(
        self,
        name: str,
        upper_limit: ndarray | float,
        standard_deviation: ndarray | float,
        field_request: FieldRequest | None = None,
        parameter_vector: ParameterVector | None = None,
        operator: ndarray | sparray | None = None,
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
        assert isinstance(self.A, (ndarray, sparray))
        assert self.A.ndim == 2
        assert self.A.shape[1] == self.n_targets
        output_shape = (self.A.shape[0],)

        self.limit = atleast_1d(upper_limit)
        self.limit = full(output_shape, self.limit) if self.limit.size == 1 else self.limit
        validate_numeric_input(
            values=self.limit,
            shape=output_shape,
            shape_name="operator output",
            error_source="SoftLimitPrior",
            input_name="upper_limit",
        )

        self.sigma = atleast_1d(standard_deviation)
        self.sigma = full(output_shape, self.sigma) if self.sigma.size == 1 else self.sigma
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


class SoftBoundsPrior(BasePrior):
    """
    A prior which is uniform between lower and upper bounds, with Gaussian
    penalties outside the interval. The same standard deviation is used for
    both tails. Equal bounds give a Gaussian penalty about that value.

    Bounds apply directly to field values or parameters, or to the output of
    an optional linear operator. Array inputs must have one element per
    target value, or per operator row when an operator is supplied.

    :param name: \
        The name used to identify the prior.

    :param lower_limit: \
        A finite, real ``float`` or one-dimensional ``numpy.ndarray`` giving
        the lower bound below which a Gaussian penalty is applied.

    :param upper_limit: \
        A finite, real ``float`` or one-dimensional ``numpy.ndarray`` giving
        the upper bound above which a Gaussian penalty is applied. Each upper
        bound must be greater than or equal to its corresponding lower bound.

    :param standard_deviation: \
        A positive, finite ``float`` or one-dimensional ``numpy.ndarray``
        giving the standard deviation of both Gaussian penalties.

    :param field_request: \
        A ``FieldRequest`` specifying the field values to which the prior applies.

    :param parameter_vector: \
        A ``ParameterVector`` specifying the parameters to which the prior applies.

    :param operator: \
        An optional finite, real, two-dimensional ``numpy.ndarray`` or
        ``scipy.sparse`` array which matrix-multiplies the target vector.
        Bounds then apply to the operator output. Sparse operators are stored
        without densifying them.
    """

    def __init__(
        self,
        name: str,
        lower_limit: ndarray | float,
        upper_limit: ndarray | float,
        standard_deviation: ndarray | float,
        field_request: FieldRequest | None = None,
        parameter_vector: ParameterVector | None = None,
        operator: ndarray | sparray | None = None,
    ):
        validate_name(name, error_source="SoftBoundsPrior")
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
                \r[ SoftBoundsPrior error ]
                \r>> One of the 'field_request' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        self.A = operator if operator is not None else eye(self.n_targets)
        if not isinstance(self.A, (ndarray, sparray)):
            raise TypeError(
                "SoftBoundsPrior 'operator' must be a numpy array or a scipy sparse array."
            )
        if self.A.ndim != 2:
            raise ValueError("SoftBoundsPrior 'operator' must be two-dimensional.")
        if self.A.shape[1] != self.n_targets:
            raise ValueError(
                "SoftBoundsPrior 'operator' must have one column for each target value."
            )
        validate_numeric_input(
            values=self.A.tocoo().data if isinstance(self.A, sparray) else self.A,
            error_source="SoftBoundsPrior",
            input_name="operator",
        )
        output_shape = (self.A.shape[0],)

        self.lower_limit = atleast_1d(lower_limit)
        if self.lower_limit.size == 1:
            self.lower_limit = full(output_shape, self.lower_limit)

        self.upper_limit = atleast_1d(upper_limit)
        if self.upper_limit.size == 1:
            self.upper_limit = full(output_shape, self.upper_limit)

        for input_name, values in (
            ("lower_limit", self.lower_limit),
            ("upper_limit", self.upper_limit),
        ):
            validate_numeric_input(
                values=values,
                shape=output_shape,
                shape_name="operator output",
                error_source="SoftBoundsPrior",
                input_name=input_name,
            )
        if (self.lower_limit > self.upper_limit).any():
            raise ValueError(
                "SoftBoundsPrior 'lower_limit' must not exceed 'upper_limit'."
            )

        self.sigma = atleast_1d(standard_deviation)
        self.sigma = full(output_shape, self.sigma) if self.sigma.size == 1 else self.sigma
        validate_numeric_input(
            values=self.sigma,
            shape=output_shape,
            shape_name="operator output",
            error_source="SoftBoundsPrior",
            input_name="standard_deviation",
            limits=(0.0, float("inf")),
            strict_limits=True,
        )
        self.weight = 1.0 / self.sigma**2

    def probability(self, **kwargs: ndarray) -> float:
        values = self.A @ kwargs[self.target]
        residual = maximum(values - self.upper_limit, 0.0) - maximum(
            self.lower_limit - values, 0.0
        )
        return -0.5 * (self.weight * residual**2).sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        values = self.A @ kwargs[self.target]
        residual = maximum(values - self.upper_limit, 0.0) - maximum(
            self.lower_limit - values, 0.0
        )
        return {self.target: -self.A.T @ (self.weight * residual)}


