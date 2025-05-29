from dataclasses import dataclass
from numpy import ndarray


@dataclass
class ParameterVector:
    name: str
    size: int

    def __post_init__(self):
        assert isinstance(self.size, int)
        assert self.size > 0
        assert isinstance(self.name, str)
        assert len(self.name) > 0


@dataclass
class FieldRequest:
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
        coord_key = tuple((name, arr.tobytes()) for name, arr in self.coordinates.items())
        self.__hash = hash((self.name, coord_key))

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return self.__hash == hash(other)