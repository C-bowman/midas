from dataclasses import dataclass
from numpy import ndarray


@dataclass
class ParameterVector:
    """
    A class used for specifying the parameters required to evaluate a
    diagnostic model, field model or prior distribution.

    :param: name \
        The name of the parameter(s).

    :param size: \
        The size of the parameter set.
    """
    name: str
    size: int

    def __post_init__(self):
        assert isinstance(self.size, int)
        assert self.size > 0
        assert isinstance(self.name, str)
        assert len(self.name) > 0


@dataclass
class FieldRequest:
    """
    A class used to request the values of particular fields which
    are required to evaluate a diagnostic model or prior distribution.

    :param name: \
        The name of the field from which values are being requested.

    :param coordinates: \
        The coordinates at which the field values are being requested as
        a dictionary mapping the coordinate names to 1D arrays of coordinate
        values.
    """
    name: str
    coordinates: dict[str, ndarray]

    def __post_init__(self):
        # validate the inputs
        assert isinstance(self.name, str)
        assert isinstance(self.coordinates, dict)
        coord_sizes = set()
        for key, value in self.coordinates.items():
            assert isinstance(key, str)
            assert isinstance(value, ndarray)
            assert value.ndim == 1
            coord_sizes.add(value.size)
        # if set size is 1, then all coord arrays are of equal size
        assert len(coord_sizes) == 1
        self.size = coord_sizes.pop()
        # converting coordinate numpy array data to bytes allows us to create
        # a hashable key for the overall coordinate set
        coord_key = tuple((name, arr.tobytes()) for name, arr in self.coordinates.items())
        # use a tuple of the field name and coordinate key to create a key for
        # the field request.
        self.__hash = hash((self.name, coord_key))

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return self.__hash == hash(other)