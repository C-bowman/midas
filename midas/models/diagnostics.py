from abc import ABC, abstractmethod
from numpy import atleast_1d, ndarray
from midas import Parameters, Fields, FieldRequest
from midas.types import Pullback


class DiagnosticModel(ABC):
    """
    An abstract base-class for diagnostic models.
    """
    parameters: Parameters
    fields: Fields

    @abstractmethod
    def predictions(self, **parameters_and_fields: ndarray) -> ndarray:
        """
        Calculate the model predictions of the measured diagnostic data.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``fields``
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
            ``FieldRequest`` objects stored in ``parameters`` and ``fields``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The model predictions of the measured diagnostic data as a 1D array,
            followed by the Jacobians of the predictions with respect to the given
            parameter and field values.

            The Jacobians must be returned as a dictionary mapping each parameter or
            field name to its corresponding Jacobian. A 1D array may be returned for
            a scalar parameter or field input. For vector inputs, the Jacobian must
            have shape ``(n_predictions, n_input_values)``.

            These must be partial derivatives with all other inputs held fixed.
            Contributions through fields that depend on a requested parameter are
            propagated separately and added to its direct contribution.
        """

    def predictions_and_pullback(
        self, **parameters_and_fields: ndarray
    ) -> tuple[ndarray, Pullback]:
        """
        Calculate the model predictions and return a function which propagates a
        gradient with respect to those predictions back to the model inputs.

        The default implementation constructs the Jacobians using
        :meth:`predictions_and_jacobians`. Models may override this method to compute
        vector-Jacobian products directly without constructing full Jacobian arrays.

        :param parameters_and_fields: \
            The parameter and field values requested by the model.

        :return: \
            The model predictions followed by a pullback function. Given the gradient
            of a scalar quantity with respect to the predictions, the pullback returns
            its gradients with respect to each requested parameter and field as 1D
            arrays.
        """
        predictions, jacobians = self.predictions_and_jacobians(
            **parameters_and_fields
        )

        def pullback(vector: ndarray) -> dict[str, ndarray]:
            return {
                name: atleast_1d(vector @ jacobian)
                for name, jacobian in jacobians.items()
            }

        return predictions, pullback


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
        self.fields = Fields(field)
        self.field_name = field.name
        self.A = model_matrix
        self.jacobian = {self.field_name: self.A}

    def predictions(self, **kwargs: ndarray) -> ndarray:
        return self.A @ kwargs[self.field_name]

    def predictions_and_jacobians(
        self, **kwargs: ndarray
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return self.A @ kwargs[self.field_name], self.jacobian
