from typing import Callable, TypeAlias
from numpy import ndarray

Coordinates: TypeAlias = dict[str, ndarray]
Pullback: TypeAlias = Callable[[ndarray], dict[str, ndarray]]