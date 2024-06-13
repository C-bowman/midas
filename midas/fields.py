from abc import ABC, abstractmethod
from numpy import ndarray, zeros, diff
from midas.parameters import FieldRequest


class FieldModel(ABC):
    n_params: int
    name: str

    @abstractmethod
    def get_values(self, parameters: ndarray, field: FieldRequest):
        pass

    @abstractmethod
    def get_values_and_jacobian(self, parameters: ndarray, field: FieldRequest):
        pass


class PiecewiseLinearField(FieldModel):
    def __init__(self, field_name: str, axis: ndarray, axis_name: str):
        assert axis.ndim == 1
        assert axis.size > 1
        assert (diff(axis) > 0.).all()
        self.name = field_name
        self.n_params = axis.size
        self.axis = axis
        self.axis_name = axis_name
        self.matrix_cache = {}

    def get_basis(self, field: FieldRequest):
        if field in self.matrix_cache:
            A = self.matrix_cache[field]
        else:
            A = self.build_linear_basis(
                x=field.coordinates[self.axis_name],
                knots=self.axis
            )
            self.matrix_cache[field] = A
        return A

    def get_values(self, parameters: ndarray, field: FieldRequest):
        basis = self.get_basis(field)
        return basis @ parameters

    def get_values_and_jacobian(self, parameters: ndarray, field: FieldRequest):
        basis = self.get_basis(field)
        return basis @ parameters, basis

    def build_linear_basis(self, x, knots):
        basis = zeros([x.size, knots.size])
        for i in range(knots.size - 1):
            k = ((x >= knots[i]) & (x <= knots[i + 1])).nonzero()
            basis[k, i + 1] = (x[k] - knots[i]) / (knots[i + 1] - knots[i])
            basis[k, i] = 1 - basis[k, i + 1]
        return basis
