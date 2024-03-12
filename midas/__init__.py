from numpy import ndarray
from dataclasses import dataclass
from abc import ABC, abstractmethod


class FieldParameterisation(ABC):
    n_params: int

    @abstractmethod
    def build_interpolator_matrix(self, *args) -> ndarray:
        pass


class PlasmaState:
    """
    this 'Plasma' object could serve as the interface to the plasma solution,
    and would contain the TriangularMesh object or some other representation
    of the plasma solution
    """

    theta: ndarray
    radius: ndarray
    n_params: int
    slices = {}
    parameters = {}
    fields: FieldParameterisation

    @classmethod
    def specify_field_model(cls, field_model: FieldParameterisation):
        cls.fields = field_model

    @classmethod
    def build_parametrisation(cls, components):
        # inspect all the posterior components and extract all fields
        # and parameters that are required
        assert cls.fields is not None
        cls.parameters = {}
        for c in components:
            for p in c.parameters:
                if p.tag not in cls.parameters:
                    cls.parameters[p.tag] = p
                elif cls.parameters[p.tag] != p:
                    raise ValueError(
                        f"""
                        Two instances of 'Parameter' have matching tags but differ in
                        type or size:
                        >> types are '{p.type}' and '{cls.parameters[p.tag].type}'
                        >> sizes are '{p.size}' and '{cls.parameters[p.tag].size}'
                        """
                    )

        # sort by type, then by variable name
        pars = sorted(cls.parameters.values(), key=lambda x: x.tag)
        pars = sorted(pars, key=lambda x: x.type)
        # replace the types with the variable sizes
        # now build pairs of parameter names and slice objects
        slices = []
        for p in pars:
            if len(slices) == 0:
                slices.append((p.tag, slice(0, p.size)))
            else:
                last = slices[-1][1].stop
                slices.append((p.tag, slice(last, last + p.size)))

        # the stop field of the last slice is the total number of parameter values
        cls.n_params = slices[-1][1].stop
        # convert to a dictionary which maps parameter names to corresponding
        # slices of the parameter vector
        cls.slices = dict(slices)

    @classmethod
    def get(cls, tag):
        return cls.theta[cls.slices[tag]]

    @classmethod
    def values_and_slice(cls, tag):
        slc = cls.slices[tag]
        return cls.theta[slc], slc

    def split_parameters(self, theta):
        return {tag: theta[slc] for tag, slc in self.slices.items()}


class Posterior:
    def __init__(self, components: list):
        self.components = components
        PlasmaState.build_parametrisation(components)

    def __call__(self, theta: ndarray) -> float:
        PlasmaState.theta = theta.copy()
        return sum(comp() for comp in self.components)

    def gradient(self, theta: ndarray) -> ndarray:
        PlasmaState.theta = theta.copy()
        return sum(comp.gradient() for comp in self.components)

    def cost(self, theta: ndarray) -> float:
        return -self.__call__(theta)

    def cost_gradient(self, theta: ndarray) -> ndarray:
        return -self.gradient(theta)


@dataclass(eq=True)
class Parameter:
    tag: str
    type: str
    size: int
