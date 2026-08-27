from numpy import ndarray, sqrt, arctan2
from scipy.interpolate import RectBivariateSpline
from abc import ABC, abstractmethod
from midas.parameters import Coordinates


class CoordinateTransform(ABC):
    """
    An abstract base class for transformations between coordinate systems.

    Subclasses declare the required coordinate names in ``inputs`` and the generated
    coordinate names in ``outputs``.
    """

    inputs: tuple[str]
    outputs: tuple[str]

    @abstractmethod
    def __call__(self, input_coords: Coordinates) -> Coordinates:
        pass


class PsiTransform(CoordinateTransform):
    """
    Transform cylindrical ``R`` and ``z`` coordinates to a poloidal-flux coordinate.

    :param R: \
        The 1D major-radius grid on which ``psi`` is defined.

    :param z: \
        The 1D vertical-coordinate grid on which ``psi`` is defined.

    :param psi: \
        The poloidal-flux values on the ``(R, z)`` grid as a 2D array with shape
        ``(R.size, z.size)``.
    """

    inputs = ("R", "z")
    outputs = ("psi",)

    def __init__(self, R: ndarray, z: ndarray, psi: ndarray):
        assert all(isinstance(arr, ndarray) for arr in [R, z, psi])
        assert R.ndim == z.ndim == 1
        assert psi.ndim == 2
        assert (R.size, z.size) == psi.shape

        self.R = R
        self.z = z
        self.psi = psi
        self.spline = RectBivariateSpline(x=R, y=z, z=psi)

    def __call__(self, coords: Coordinates) -> Coordinates:
        """
        Evaluate the poloidal flux at the supplied cylindrical coordinates.

        :param coords: \
            Coordinates containing ``"R"`` and ``"z"`` arrays with compatible
            shapes.

        :return: \
            Coordinates containing the interpolated poloidal flux under ``"psi"``.
        """
        return {"psi": self.spline(x=coords["R"], y=coords["z"], grid=False)}


class CylindricalTransform(CoordinateTransform):
    """
    Transform Cartesian ``x``, ``y``, and ``z`` coordinates to ``R``, ``z``, and
    azimuthal angle ``phi``.
    """

    inputs = ("x", "y", "z")
    outputs = ("R", "z", "phi")

    def __call__(self, coords: Coordinates) -> Coordinates:
        """
        Convert the supplied Cartesian coordinates to cylindrical coordinates.

        :param coords: \
            Coordinates containing compatible ``"x"``, ``"y"``, and ``"z"``
            arrays.

        :return: \
            Coordinates containing ``"R"``, ``"z"``, and ``"phi"`` arrays, where
            ``"phi"`` is the azimuthal angle in radians.
        """
        return {
            "R": sqrt(coords["x"] ** 2 + coords["y"] ** 2),
            "z": coords["z"],
            "phi": arctan2(coords["y"], coords["x"]),
        }
