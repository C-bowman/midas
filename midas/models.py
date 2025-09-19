from abc import ABC, abstractmethod
from numpy import ndarray
from midas.parameters import FieldRequest, ParameterVector


class DiagnosticModel(ABC):
    parameters: list[ParameterVector]
    field_requests: list[FieldRequest]

    @abstractmethod
    def predictions(self, **kwargs):
        pass

    @abstractmethod
    def predictions_and_jacobians(self, **kwargs):
        pass


class LinearDiagnosticModel(DiagnosticModel):
    def __init__(self, field: FieldRequest, model_matrix: ndarray):
        self.parameters = []
        self.field_requests = [field]
        self.field_name = field.name
        self.A = model_matrix
        self.jacobian = {self.field_name: self.A}

    def predictions(self, **kwargs):
        return self.A @ kwargs[self.field_name]

    def predictions_and_jacobians(self, **kwargs):
        return self.A @ kwargs[self.field_name], self.jacobian