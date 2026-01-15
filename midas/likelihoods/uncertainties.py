from abc import ABC, abstractmethod
from numpy import ndarray, full
from midas import Parameters


class UncertaintyModel(ABC):
    """
    An abstract base-class for uncertainty models.
    """

    size: int
    name: str
    parameters: Parameters

    @abstractmethod
    def get_uncertainties(self, parameters: dict[str, ndarray]) -> ndarray:
        """
        Get the values of the uncertainties.

        :param parameters: \
            The parameter values requested via the ``ParameterVector`` objects stored
            in the ``parameters`` instance attribute.
            These values are given as a dictionary mapping the parameter names
            (specified by the ``name`` attribute of the ``ParameterVector`` objects)
            to the parameter values as 1D arrays.

        :return: \
            The modelled uncertainty values as a 1D array.
        """
        pass

    @abstractmethod
    def get_uncertainties_and_jacobians(
        self, parameters: dict[str, ndarray]
    ) -> tuple[ndarray, dict[str, ndarray]]:
        """
        Get the values of the uncertainties, and the Jacobians of the uncertainties
        values with respect to the given parameters values.

        :param parameters: \
            The parameter values requested via the ``ParameterVector`` objects stored
            in the ``parameters`` instance attribute.
            These values are given as a dictionary mapping the parameter names
            (specified by the ``name`` attribute of the ``ParameterVector`` objects)
            to the parameter values as 1D arrays.

        :return: \
            The uncertainty values as a 1D array, followed by the Jacobians of the
            uncertainties with respect to the given parameter values.

            The Jacobians must be returned as a dictionary mapping the parameter names
            to the corresponding Jacobians as 2D arrays.
        """
        pass


class ConstantUncertainty(UncertaintyModel):
    """
    Models the certainty on a set of data given to a likelihood function as a single
    value shared by all the data.

    :param n_data: \
        The number data points for which uncertainties are being modelled.

    :param parameter_name: \
        The name of the parameter which sets the uncertainty value for all data points.
    """

    def __init__(self, n_data: int, parameter_name: str):
        self.size = n_data
        self.name = parameter_name
        self.parameters = Parameters((self.name, 1))
        self.jacobian = {self.name: full(self.size, 1.0)}

    def get_uncertainties(self, parameters: dict[str, ndarray]) -> ndarray:
        return full(self.size, parameters[self.name])

    def get_uncertainties_and_jacobians(
        self, parameters: dict[str, ndarray]
    ) -> tuple[ndarray, dict[str, ndarray]]:
        return full(self.size, parameters[self.name]), self.jacobian


class LinearUncertainty(UncertaintyModel):
    """
    Models the certainty on a set of data given to a likelihood function as a linear
    function of the data values.

    :param y_data: \
        The data values given to the likelihood function as a 1D array.

    :param parameter_prefix: \
        A prefix added to the names of the parameters specifying the fractional and
        constant components of the modelled uncertainties.
    """

    def __init__(self, y_data: ndarray, parameter_prefix: str):
        self.y_data = abs(y_data)  # absolute values to prevent negative uncertainties
        self.size = y_data.size
        self.const_name = f"{parameter_prefix}_constant_error"
        self.frac_name = f"{parameter_prefix}_fractional_error"
        self.parameters = Parameters((self.const_name, 1), (self.frac_name, 1))
        self.jacobian = {
            self.const_name: full(self.size, 1.0),
            self.frac_name: self.y_data,
        }

    def get_uncertainties(self, parameters: dict[str, ndarray]) -> ndarray:
        return parameters[self.frac_name] * self.y_data + parameters[self.const_name]

    def get_uncertainties_and_jacobians(
        self, parameters: dict[str, ndarray]
    ) -> tuple[ndarray, dict[str, ndarray]]:
        uncertainties = (
            parameters[self.frac_name] * self.y_data + parameters[self.const_name]
        )
        return uncertainties, self.jacobian
