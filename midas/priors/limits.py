from numpy import eye, ndarray, maximum
from midas.state import BasePrior
from midas.parameters import Parameters, Fields, FieldRequest


class SoftLimitPrior(BasePrior):
    def __init__(
        self,
        name: str,
        field_request: FieldRequest,
        upper_limit: float,
        sigma: float,
        operator: ndarray = None,
    ):
        self.name = name
        self.field_name = field_request.name
        self.fields = Fields(field_request)
        self.parameters = Parameters()
        self.limit = upper_limit
        self.sigma = sigma
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

