from abc import ABC, abstractmethod
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
