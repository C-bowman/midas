from numpy import ndarray, atleast_1d
from midas.parameters import ParameterVector, FieldRequest
from midas.parameters import Parameters, Fields
from midas.state import BasePrior
from midas.validation import validate_name, validate_numeric_input


class GaussianPrior(BasePrior):
    """
    Specify a Gaussian prior over either a series of field values, or a
    set of parameters.

    :param name: \
        The name used to identify the Gaussian prior.

    :param mean: \
        The mean of the Gaussian prior corresponding to each parameter or requested
        field value.

    :param standard_deviation: \
        The standard deviation of the Gaussian prior corresponding to each parameter
        or requested field value.

    :param field_request: \
        A ``FieldRequest`` specifying the field and coordinates to which the Gaussian
        prior will be applied. If specified, ``field_request`` will override
        any values passed to the ``parameter_vector`` arguments.

    :param parameter_vector: \
        A ``ParameterVector`` specifying the parameters to which the Gaussian prior
        will be applied.
    """

    def __init__(
        self,
        name: str,
        mean: ndarray,
        standard_deviation: ndarray,
        field_request: FieldRequest | None = None,
        parameter_vector: ParameterVector | None = None,
    ):
        validate_name(name, error_source="GaussianPrior")
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
                \r[ GaussianPrior error ]
                \r>> One of the 'field_request' or 'parameter_vector' keyword arguments
                \r>> must be specified with a ``FieldRequest`` or ``ParameterVector``
                \r>> object respectively.
                """
            )

        self.mean = atleast_1d(mean)
        validate_numeric_input(
            values=self.mean,
            shape=(self.n_targets,),
            shape_name=self.target_type,
            error_source="GaussianPrior",
            input_name="mean",
        )

        self.sigma = atleast_1d(standard_deviation)
        validate_numeric_input(
            values=self.sigma,
            shape=(self.n_targets,),
            shape_name=self.target_type,
            error_source="GaussianPrior",
            input_name="standard_deviation",
            limits=(0.0, float("inf")),
            strict_limits=True,
        )

        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2

    def probability(self, **kwargs: ndarray) -> float:
        target_values = kwargs[self.target]
        z = (target_values - self.mean) * self.inv_sigma
        return -0.5 * (z**2).sum()

    def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
        target_values = kwargs[self.target]
        return {self.target: -(target_values - self.mean) * self.inv_sigma_sqr}
