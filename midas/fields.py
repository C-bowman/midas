from abc import ABC, abstractmethod
from numpy import ndarray, zeros, diff
from midas.parameters import FieldRequest, ParameterVector


class FieldModel(ABC):
    n_params: int
    name: str
    parameters: list[ParameterVector]

    @abstractmethod
    def get_values(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> ndarray:
        pass

    @abstractmethod
    def get_values_and_jacobian(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> tuple[ndarray, ndarray]:
        pass


class PiecewiseLinearField(FieldModel):
    def __init__(self, field_name: str, axis: ndarray, axis_name: str):
        assert axis.ndim == 1
        assert axis.size > 1
        assert (diff(axis) > 0.0).all()
        self.name = field_name
        self.n_params = axis.size
        self.axis = axis
        self.axis_name = axis_name
        self.matrix_cache = {}
        self.param_name = f"{field_name}_linear_basis"
        self.parameters = [ParameterVector(name=self.param_name, size=self.n_params)]

    def get_basis(self, field: FieldRequest) -> ndarray:
        if field in self.matrix_cache:
            A = self.matrix_cache[field]
        else:
            A = self.build_linear_basis(
                x=field.coordinates[self.axis_name], knots=self.axis
            )
            self.matrix_cache[field] = A
        return A

    def get_values(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> ndarray:
        basis = self.get_basis(field)
        return basis @ parameters[self.param_name]

    def get_values_and_jacobian(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> tuple[ndarray, dict[str, ndarray]]:
        basis = self.get_basis(field)
        return basis @ parameters[self.param_name], {self.param_name: basis}

    @staticmethod
    def build_linear_basis(x: ndarray, knots: ndarray) -> ndarray:
        basis = zeros([x.size, knots.size])
        for i in range(knots.size - 1):
            k = ((x >= knots[i]) & (x <= knots[i + 1])).nonzero()
            basis[k, i + 1] = (x[k] - knots[i]) / (knots[i + 1] - knots[i])
            basis[k, i] = 1 - basis[k, i + 1]
        return basis
