from abc import ABC, abstractmethod
from numpy import ndarray
from midas.parameters import FieldRequest, FieldRequests, Parameters


class DiagnosticModel(ABC):
    """
    An abstract base-class for diagnostic models.
    """
    parameters: Parameters
    field_requests: FieldRequests

    @abstractmethod
    def predictions(self, **parameters_and_fields: ndarray) -> ndarray:
        """
        Calculate the model predictions of the measured diagnostic data.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``field_requests``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The model predictions of the measured diagnostic data as a 1D array.
        """
        pass

    @abstractmethod
    def predictions_and_jacobians(
        self, **parameters_and_fields: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        """
        Calculate the model predictions of the measured diagnostic data, and the
        Jacobians of the predictions with respect to the given parameter and field
        values.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``field_requests``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The model predictions of the measured diagnostic data as a 1D array,
            followed by the Jacobians of the predictions with respect to the given
            parameter and field values.

            The Jacobians must be returned as a dictionary mapping the parameter and
            field names to the corresponding Jacobians as 2D arrays.
        """


class LinearDiagnosticModel(DiagnosticModel):
    """
    A class for purely linear diagnostic models, where the model predictions are
    obtained from the product of a model matrix and a vector of field values.

    :param field: \
        A ``FieldRequest`` specifying the vector of field values which will be
        multiplied by the given ``model_matrix``.

    :param model_matrix: \
        A matrix which will multiply the vector of requested field values in
        order to produce the model predictions.
    """
    def __init__(self, field: FieldRequest, model_matrix: ndarray):
        self.parameters = Parameters()
        self.field_requests = FieldRequests(field)
        self.field_name = field.name
        self.A = model_matrix
        self.jacobian = {self.field_name: self.A}

    def predictions(self, **kwargs: ndarray) -> ndarray:
        return self.A @ kwargs[self.field_name]

    def predictions_and_jacobians(
        self, **kwargs: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return self.A @ kwargs[self.field_name], self.jacobian
