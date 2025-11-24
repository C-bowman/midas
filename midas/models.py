from abc import ABC, abstractmethod
from numpy import ndarray
from midas.parameters import FieldRequest, ParameterVector


class DiagnosticModel(ABC):
    parameters: list[ParameterVector]
    field_requests: list[FieldRequest]

    @abstractmethod
    def predictions(
        self, parameters: dict[str, ndarray], fields: dict[str, ndarray]
    ) -> ndarray:
        """
        Calculate the model predictions of the measured diagnostic data.

        :param parameters: \
            The parameter values requested via the ``ParameterVector`` objects stored
            in the ``parameters`` instance attribute.
            These values are given as a dictionary mapping the parameter names
            (specified by the ``name`` attribute of the ``ParameterVector`` objects)
            to the parameter values as 1D arrays.

        :param fields: \
            The field values requested via the ``FieldRequest`` objects stored
            in the ``field_requests`` instance attribute.
            These values are given as a dictionary mapping the field names
            (specified by the ``name`` attribute of the ``FieldRequest`` objects)
            to the field values as 1D arrays.

        :return: \
            The model predictions of the measured diagnostic data as a 1D array.
        """
        pass

    @abstractmethod
    def predictions_and_jacobians(
        self, parameters: dict[str, ndarray], fields: dict[str, ndarray]
    ) -> tuple[ndarray, dict[str, ndarray]]:
        """
        Calculate the model predictions of the measured diagnostic data,
        and the Jacobians of the predictions with respect to the given
        parameter and field values.

        :param parameters: \
            The parameter values requested via the ``ParameterVector`` objects stored
            in the ``parameters`` instance attribute.
            These values are given as a dictionary mapping the parameter names
            (specified by the ``name`` attribute of the ``ParameterVector`` objects)
            to the parameter values as 1D arrays.

        :param fields: \
            The field values requested via the ``FieldRequest`` objects stored
            in the ``field_requests`` instance attribute.
            These values are given as a dictionary mapping the field names
            (specified by the ``name`` attribute of the ``FieldRequest`` objects)
            to the field values as 1D arrays.

        :return: \
            The model predictions of the measured diagnostic data as a 1D array,
            followed by the Jacobians of the predictions with respect to the given
            parameter and field values.

            The Jacobians must be returned as a dictionary mapping the parameter and
            field names to the corresponding Jacobians as 2D arrays.
        """
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
