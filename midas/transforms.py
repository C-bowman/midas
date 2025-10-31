from numpy import ndarray, sqrt, arctan2
from scipy.interpolate import RectBivariateSpline
from abc import ABC, abstractmethod

coordinates = dict[str, ndarray]


class CoordinateTransform(ABC):
    inputs: tuple[str]
    outputs: tuple[str]

    @abstractmethod
    def __call__(self, input_coords: coordinates) -> coordinates:
        pass


class PsiTransform(CoordinateTransform):
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

    def __call__(self, coords: coordinates) -> coordinates:
        return {"psi": self.spline(x=coords["R"], y=coords["z"], grid=False)}


class CylindricalTransform(CoordinateTransform):
    inputs = ("x", "y", "z")
    outputs = ("R", "z", "phi")

    def __call__(self, coords: coordinates) -> coordinates:
        return {
            "R": sqrt(coords["x"] ** 2 + coords["y"] ** 2),
            "z": coords["z"],
            "phi": arctan2(coords["y"], coords["x"]),
        }
