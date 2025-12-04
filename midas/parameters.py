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


class Parameters(tuple):
    """
    A tuple subclass which creates an immutable collection of validated ``ParameterVector``
    objects. The arguments should be a series of ``ParameterVector``.
    """
    def __new__(cls, *parameters: ParameterVector):
        parameter_names = set()
        for param in parameters:
            if not isinstance(param, ParameterVector):
                raise TypeError(
                    f"""\n
                    \r[ Parameters error ]
                    \r>> All arguments passed to Parameters must have type
                    \r>> ``ParameterVector``, but instead an argument has type:
                    \r>> {type(param)}
                    """
                )

            # check that all the parameter names in the current prior are unique
            if param.name not in parameter_names:
                parameter_names.add(param.name)
            else:
                raise ValueError(
                    f"""\n
                    \r[ Parameters error ]
                    \r>> At least two given ``ParameterVector`` objects share the name:
                    \r>> '{param.name}'
                    \r>> but all names must be unique.
                    """
                )

        return tuple.__new__(cls, parameters)


class Fields(tuple):
    """
    A tuple subclass which creates an immutable collection of validated ``FieldRequest``
    objects. The arguments should be a series of ``FieldRequest``.
    """
    def __new__(cls, *field_requests: FieldRequest):
        field_names = set()
        for request in field_requests:
            if not isinstance(request, FieldRequest):
                raise TypeError(
                    f"""\n
                    \r[ FieldRequests error ]
                    \r>> All arguments passed to FieldRequests must have type
                    \r>> ``FieldRequest``, but instead an argument has type:
                    \r>> {type(request)}
                    """
                )

            # check that all the parameter names in the current prior are unique
            if request.name not in field_names:
                field_names.add(request.name)
            else:
                raise ValueError(
                    f"""\n
                    \r[ FieldRequests error ]
                    \r>> At least two given ``ParameterVector`` objects share the name:
                    \r>> '{request.name}'
                    \r>> but all names must be unique.
                    """
                )

        return tuple.__new__(cls, field_requests)


def validate_parameters(model, error_source: str, description: str):
    valid_parameters = (
        hasattr(model, "parameters")
        and isinstance(model.parameters, Parameters)
    )
    if not valid_parameters:
        raise TypeError(
            f"""\n
            \r[ {error_source} error ]
            \r>> The {description}
            \r>> does not possess a valid 'parameters' instance attribute.
            \r>> 'parameters' must be an instance of the ``Parameters`` class.
            """
        )

def validate_field_requests(model, error_source: str, description: str):
    valid_field_requests = (
        hasattr(model, "fields")
        and isinstance(model.fields, Fields)
    )
    if not valid_field_requests:
        raise TypeError(
            f"""\n
            \r[ {error_source} error ]
            \r>> The {description}
            \r>> does not possess a valid 'fields' instance attribute.
            \r>> 'fields' must be a instance of the ``Fields`` class.
            """
        )
