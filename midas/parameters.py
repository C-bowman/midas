from dataclasses import dataclass, field
from numpy import ndarray


@dataclass
class ParameterVector:
    name: str
    size: int


@dataclass
class FieldRequest:
    name: str
    coordinates: dict[str, ndarray]

    def __post_init__(self):
        key = tuple((name, arr.tobytes()) for name, arr in self.coordinates.items())
        self.__hash = hash(key)

    def __hash__(self):
        return self.__hash
