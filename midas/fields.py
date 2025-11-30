from abc import ABC, abstractmethod
from numpy import ndarray, zeros, diff
from midas.parameters import FieldRequest, ParameterVector, Parameters


class FieldModel(ABC):
    """
    An abstract base-class for field models.
    """
    n_params: int
    name: str
    parameters: Parameters

    @abstractmethod
    def get_values(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> ndarray:
        """
        Get the values of the field at a set of given coordinates.

        :param parameters: \
            The parameter values requested via the ``ParameterVector`` objects stored
            in the ``parameters`` instance attribute.
            These values are given as a dictionary mapping the parameter names
            (specified by the ``name`` attribute of the ``ParameterVector`` objects)
            to the parameter values as 1D arrays.

        :param field: \
            A ``FieldRequest`` specifying the coordinates at which the modelled field
            values should be calculated.

        :return: \
            The modelled field values as a 1D array.
        """
        pass

    @abstractmethod
    def get_values_and_jacobian(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> tuple[ndarray, dict[str, ndarray]]:
        """
        Get the values of the field at a set of given coordinates, and the Jacobian
        of those fields values with respect to the given parameters values.

        :param parameters: \
            The parameter values requested via the ``ParameterVector`` objects stored
            in the ``parameters`` instance attribute.
            These values are given as a dictionary mapping the parameter names
            (specified by the ``name`` attribute of the ``ParameterVector`` objects)
            to the parameter values as 1D arrays.

        :param field: \
            A ``FieldRequest`` specifying the coordinates at which the modelled field
            values should be calculated.

        :return: \
            The field values as a 1D array, followed by the Jacobians of the field
            values with respect to the given parameter values.

            The Jacobians must be returned as a dictionary mapping the parameter names
            to the corresponding Jacobians as 2D arrays.
        """
        pass


class PiecewiseLinearField(FieldModel):
    """
    Models a chosen field as a piecewise-linear 1D profile.

    :param field_name: \
        The name of the field to be modelled.

    :param axis: \
        Coordinate values specifying the locations of the basis functions
        which make up the 1D profile. The number of free parameters of the
        field model will be equal to the size of ``axis``. The coordinate
        values must be given in strictly ascending order.

    :param axis_name: \
        The name of the coordinate over which the 1D profile is defined.
    """
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
        self.parameters = Parameters(
            ParameterVector(name=self.param_name, size=self.n_params)
        )

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
