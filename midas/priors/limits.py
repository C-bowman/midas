from numpy import eye, ndarray, maximum
from midas.state import BasePrior
from midas.parameters import Parameters, Fields, FieldRequest


class SoftLimitPrior(BasePrior):
    """
    A prior which is uniform up to a given upper-limit, and beyond which is
    a Gaussian with a given standard deviation. This allows a 'soft' limit
    to be imposed on a set of given field values.

    Alternatively, a 2D matrix operator can also be given, and the prior will instead
    be applied to the result of the matrix multiplication of that operator and the
    vector of requested field-values.

    :param name: \
        The name used to identify the prior.

    :param upper_limit: \
        The upper limit beyond which a Gaussian prior is applied to the field values.

    :param standard_deviation: \
        The standard deviation of the Gaussian prior applied to each field value which is
        above the given upper limit.

    :param field_request: \
        A ``FieldRequest`` specifying the field values to which the prior is applied.

    :param operator: \
        A linear operator (as a 2D array) which matrix-multiplies the vector of
        requested field values. If specified, the prior will be applied to the result
        of this matrix-multiplication instead of the requested field values.
    """

    def __init__(
        self,
        name: str,
        upper_limit: float,
        standard_deviation: float,
        field_request: FieldRequest,
        operator: ndarray = None,
    ):
        self.name = name
        self.field_name = field_request.name
        self.fields = Fields(field_request)
        self.parameters = Parameters()
        self.limit = upper_limit
        self.sigma = standard_deviation
        self.weight = 1.0 / self.sigma**2
        self.A = operator if operator is not None else eye(field_request.size)
        assert self.A.ndim == 2
        assert self.A.shape[1] == field_request.size

    def probability(self, **fields):
        v = fields[self.field_name]
        z = self.A @ v - self.limit
        return -0.5 * self.weight * (maximum(z, 0.0) ** 2).sum()

    def gradients(self, **fields):
        v = fields[self.field_name]
        z = self.A @ v - self.limit
        return {self.field_name: -self.weight * self.A.T @ maximum(z, 0.0)}
