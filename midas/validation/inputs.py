from numpy import ndarray, issubdtype, integer, floating, isfinite


def validate_numeric_input(
    values: ndarray,
    error_source: str,
    input_name: str,
    shape: tuple[int, ...] | None = None,
    shape_name: str | None = None,
    limits: tuple[float, float] | None = None,
    strict_limits: bool = False,
) -> None:

    if shape is not None:
        if values.ndim != len(shape):
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> '{input_name}' input must be an array with dimension {len(shape)},
                \r>> but instead has dimension {values.ndim}.
                """
            )
        
        if values.shape != shape:
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> '{input_name}' input must have shape matching the given '{shape_name}'.
                \r>> {shape_name} has shape {shape}, however
                \r>> {input_name} has shape {values.shape}.
                """
            )
    
    numeric_dtype = issubdtype(values.dtype, integer) or issubdtype(
        values.dtype, floating
    )
    if not numeric_dtype:
        raise TypeError(
            f"""\n
            \r[ {error_source} error ]
            \r>>'{input_name}' input must contain real numeric values.
            """
        )

    if not isfinite(values).all():
        raise ValueError(
            f"""\n
            \r[ {error_source} error ]
            \r>> '{input_name}' input must contain only finite values.
            """
        )

    if limits is not None:
        lower, upper = limits
        if strict_limits:
            within_limits = (values > lower) & (values < upper)
            comparison = "<"
        else:
            within_limits = (values >= lower) & (values <= upper)
            comparison = "<="
        if not within_limits.all():
            raise ValueError(
                f"""\n
                \r[ {error_source} error ]
                \r>> '{input_name}' input must contain values within the limits:
                \r>> {limits[0]} {comparison} values {comparison} {limits[1]}.
                """
            )


def validate_name(
    name: str,
    error_source: str
) -> None:
    if not isinstance(name, str):
        raise TypeError(
            f"""\n
            \r[ {error_source} error ]
            \r>> Name given to {error_source} must be a string.
            """
        )
    if not name:
        raise ValueError(
            f"""\n
            \r[ {error_source} error ]
            \r>> Name given to {error_source} must not be empty.
            """
        )