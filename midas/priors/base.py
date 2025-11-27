from abc import ABC, abstractmethod
from numpy import ndarray, zeros
from midas.state import PlasmaState
from midas.parameters import ParameterVector, FieldRequest


class BasePrior(ABC):
    parameters: list[ParameterVector]
    field_requests: list[FieldRequest]
    name: str

    @abstractmethod
    def probability(self, **parameters_and_fields: ndarray) -> float:
        """
        Calculate the prior log-probability.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``field_requests``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The prior log-probability value.
        """
        pass

    @abstractmethod
    def gradients(self, **parameters_and_fields: ndarray) -> dict[str, ndarray]:
        """
        Calculate the prior log-probability.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``field_requests``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The gradient of the prior log-probability with respect to the given
            parameter and field values. These gradients are returned as a dictionary
            mapping the parameter and field names to their respective gradients as
            1D arrays.
        """

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
