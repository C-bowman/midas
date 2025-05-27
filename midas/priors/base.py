from abc import ABC, abstractmethod
from numpy import ndarray, zeros
from midas.state import PlasmaState
from midas.parameters import ParameterVector, FieldRequest


class BasePrior(ABC):
    parameters: list[ParameterVector]
    field_requests: list[FieldRequest]
    name: str

    @abstractmethod
    def probability(self, **kwargs) -> float:
        pass

    @abstractmethod
    def gradients(self, **kwargs) -> dict[str, ndarray]:
        pass

    def log_probability(self) -> float:
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        return self.probability(**param_values, **field_values)

    def log_probability_gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = (
            PlasmaState.get_values_and_jacobians(
                parameters=self.parameters, field_requests=self.field_requests
            )
        )

        gradients = self.gradients(
            **param_values, **field_values
        )

        grad = zeros(PlasmaState.n_params)
        for p in param_values.keys():
            slc = PlasmaState.slices[p]
            grad[slc] = gradients[p]

        for field_param in field_jacobians.keys():
            field_name = PlasmaState.field_parameter_map[field_param]
            slc = PlasmaState.slices[field_param]
            grad[slc] = gradients[field_name] @ field_jacobians[field_param]

        return grad
