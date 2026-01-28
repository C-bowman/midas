from abc import ABC, abstractmethod
from numpy import arange, concatenate, diff, ndarray, zeros, exp
from scipy.linalg import solve
from tokamesh.mesh import TriangularMesh
from midas.parameters import FieldRequest, ParameterVector, Parameters
from midas.parameters import validate_coordinates


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

        self.build_basis_matrix = piecewise_linear_basis
        self.param_name = f"{field_name}_linear_basis"
        self.parameters = Parameters(
            ParameterVector(name=self.param_name, size=self.n_params)
        )

    def get_basis(self, field: FieldRequest) -> ndarray:
        if field in self.matrix_cache:
            A = self.matrix_cache[field]
        else:
            A = self.build_basis_matrix(
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


class CubicSplineField(PiecewiseLinearField):
    def __init__(self, field_name: str, axis: ndarray, axis_name: str):
        super().__init__(field_name, axis, axis_name)

        self.build_basis_matrix = cubic_spline_basis
        self.param_name = f"{field_name}_cubic_spline"
        self.parameters = Parameters(
            ParameterVector(name=self.param_name, size=self.n_params)
        )


class BSplineField(PiecewiseLinearField):
    def __init__(self, field_name: str, axis: ndarray, axis_name: str):
        super().__init__(field_name, axis, axis_name)

        self.build_basis_matrix = b_spline_basis
        self.param_name = f"{field_name}_bspline_basis"
        self.parameters = Parameters(
            ParameterVector(name=self.param_name, size=self.n_params)
        )


class ExSplineField(PiecewiseLinearField):
    def __init__(self, field_name: str, axis: ndarray, axis_name: str):
        super().__init__(field_name, axis, axis_name)

        self.build_basis_matrix = b_spline_basis
        self.param_name = f"ln_{field_name}_bspline_basis"
        self.parameters = Parameters(
            ParameterVector(name=self.param_name, size=self.n_params)
        )

    def get_values(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> ndarray:
        basis = self.get_basis(field)
        return exp(basis @ parameters[self.param_name])

    def get_values_and_jacobian(
        self, parameters: dict[str, ndarray], field: FieldRequest
    ) -> tuple[ndarray, dict[str, ndarray]]:
        basis = self.get_basis(field)
        values = exp(basis @ parameters[self.param_name])
        return values, {self.param_name: basis * values[:, None]}


class TriangularMeshField(FieldModel):
    """
    Models a chosen field using a 2D triangular mesh.

    The parameters of the field model are the field values at the mesh vertices.
    Inside each triangle of the mesh, the field is represented as the unique plane which
    passes through the 3D coordinates (two spatial coordinates plus the field value) of
    each of the triangles vertices. Consequently, the value of the field is continuous
    and well-defined everywhere inside the mesh.

    :param field_name: \
        The name of the field to be modelled.

    :param mesh_coordinates: \
        The coordinates specifying the mesh vertices as a dictionary mapping the
        names of each of the two coordinates to 1D numpy arrays of the coordinate values.

    :param triangle_vertices: \
        The indices specifying the vertices which make up each triangle in the mesh.
        This should be specified as a numpy array of integers of shape
        ``(num_triangles, 3)``.
    """
    def __init__(
        self,
        field_name: str,
        mesh_coordinates: dict[str, ndarray],
        triangle_vertices: ndarray,
    ):
        validate_coordinates(mesh_coordinates, error_source="TriangularMeshField")
        assert len(mesh_coordinates) == 2

        self.mesh_coords = [key for key in mesh_coordinates.keys()]
        self.n_params = mesh_coordinates[self.mesh_coords[0]].size

        self.mesh = TriangularMesh(
            R=mesh_coordinates[self.mesh_coords[0]],
            z=mesh_coordinates[self.mesh_coords[1]],
            triangles=triangle_vertices,
        )

        self.name = field_name
        self.matrix_cache = {}
        self.param_name = f"{field_name}_triangular_basis"
        self.parameters = Parameters(
            (self.param_name, self.n_params)
        )

    def get_basis(self, field: FieldRequest) -> ndarray:
        if field in self.matrix_cache:
            A = self.matrix_cache[field]
        else:
            A = self.mesh.build_interpolator_matrix(
                R=field.coordinates[self.mesh_coords[0]],
                z=field.coordinates[self.mesh_coords[1]],
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


def piecewise_linear_basis(x: ndarray, knots: ndarray) -> ndarray:
    basis = zeros([x.size, knots.size])
    for i in range(knots.size - 1):
        k = ((x >= knots[i]) & (x <= knots[i + 1])).nonzero()
        basis[k, i + 1] = (x[k] - knots[i]) / (knots[i + 1] - knots[i])
        basis[k, i] = 1 - basis[k, i + 1]
    return basis


def b_spline_basis(x: ndarray, knots: ndarray, order=3, derivatives=False) -> ndarray:
    assert order % 2 == 1
    iters = order + 1
    t = knots.copy()
    # we need n = order points of padding either side
    dl, dr = t[1] - t[0], t[-1] - t[-2]
    L_pad = t[0] - dl * arange(1, iters)[::-1]
    R_pad = t[-1] + dr * arange(1, iters)

    t = concatenate([L_pad, t, R_pad])
    n_knots = t.size

    # construct zeroth-order
    splines = zeros([x.size, n_knots, iters])
    for i in range(n_knots - 1):
        bools = (t[i] <= x) & (t[i + 1] > x)
        splines[bools, i, 0] = 1.

    dx = x[:, None] - t[None, :]
    for k in range(1, iters):
        dt = t[k:] - t[:-k]
        S = splines[:, :-k, k-1] / dt[None, :]
        splines[:, :-(k+1), k] = S[:, :-1] * dx[:, :-(k+1)] - S[:, 1:] * dx[:, k+1:]

    # remove the excess functions which don't contribute to the supported range
    basis = splines[:, :-iters, -1]

    # combine the functions at the edge of the supported range
    if iters // 2 > 1:
        n = (iters // 2) - 1
        basis[:, n] += basis[:, :n].sum(axis=1)
        basis[:, -(n+1)] += basis[:, -n:].sum(axis=1)
        basis = basis[:, n:-n]

    if derivatives:
        # derivative of order k splines are a weighted difference of order k-1 splines
        coeffs = order / (t[:-order] - t[order:])
        derivs = diff(splines[:, :-order, -2] * coeffs[None, :])

        if iters // 2 > 1:
            derivs[:, n] += derivs[:, :n].sum(axis=1)
            derivs[:, -(n+1)] += derivs[:, -n:].sum(axis=1)
            derivs = derivs[:, n:-n]

        return basis, derivs
    else:
        return basis


def cubic_spline_basis(x: ndarray, knots: ndarray) -> ndarray:
    B = b_spline_basis(knots, knots)
    return b_spline_basis(x, knots) @ solve(B.T @ B, B.T)
