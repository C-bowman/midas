from numpy import ndarray, zeros
from midas.fields import FieldModel
from midas.parameters import ParameterVector, FieldRequest


class PlasmaState:
    theta: ndarray
    radius: ndarray
    n_params: int
    parameter_names: set[str]
    slices: dict[str, slice] = {}
    fields: dict[str, FieldModel]

    @classmethod
    def specify_field_models(cls, field_models: list[FieldModel]):
        cls.fields = {f.name: f for f in field_models}

    @classmethod
    def build_parametrisation(cls, components: list):
        # First check that the requested fields and the modelled fields match each other
        assert cls.fields is not None
        requested_fields = set()
        [[requested_fields.add(f.name) for f in c.field_requests] for c in components]
        modelled_fields = {f for f in cls.fields.keys()}
        if modelled_fields != requested_fields:
            raise ValueError(
                f"""\n
                \r[ PlasmaState error ]
                \r>> The set of fields requested by the diagnostic likelihoods and / or
                \r>> priors does not match the set of modelled fields.
                \r>> The requested fields are:
                \r>> {requested_fields}
                \r>> but the modelled fields are:
                \r>> {modelled_fields}
                """
            )

        # sort the field sizes by name
        slice_sizes = sorted([(name, f.n_params) for name, f in cls.fields.items()])

        parameter_sizes = {}
        for c in components:
            for p in c.parameters:
                assert isinstance(p, ParameterVector)
                if p.name not in parameter_sizes:
                    parameter_sizes[p.name] = p.size
                elif parameter_sizes[p.name] != p.size:
                    raise ValueError(
                        f"""\n
                        \r[ PlasmaState error ]
                        \r>> Two instances of 'ParameterVector' have matching names '{p.name}'
                        \r>> but differ in their size:
                        \r>> sizes are '{p.size}' and '{parameter_sizes[p.name]}'
                        """
                    )

        # sort the parameter sizes by name
        slice_sizes.extend(
            sorted([t for t in parameter_sizes.items()], key=lambda x: x[0])
        )
        # now build pairs of parameter names and slice objects
        slices = []
        for name, size in slice_sizes:
            if len(slices) == 0:
                slices.append((name, slice(0, size)))
            else:
                last = slices[-1][1].stop
                slices.append((name, slice(last, last + size)))

        # the stop field of the last slice is the total number of parameters
        cls.n_params = slices[-1][1].stop
        # convert to a dictionary which maps parameter names to corresponding
        # slices of the parameter vector
        cls.slices = dict(slices)
        cls.parameter_names = {name for name in cls.slices.keys()}

    @classmethod
    def split_parameters(cls, theta: ndarray) -> dict[str, ndarray]:
        if not isinstance(theta, ndarray) or theta.shape != (cls.n_params,):
            raise ValueError(
                f"""\n
                \r[ PlasmaState.split_parameters error ]
                \r>> Given 'theta' argument must be an instance of a
                \r>> numpy.ndarray with shape ({cls.n_params},).
                """
            )
        return {tag: theta[slc] for tag, slc in cls.slices.items()}

    @classmethod
    def merge_parameters(cls, parameter_values: dict[str, ndarray | float]) -> ndarray:
        theta = zeros(cls.n_params)

        missing_params = cls.parameter_names - {k for k in parameter_values.keys()}
        if len(missing_params) > 0:
            raise ValueError(
                f"""\n
                \r[ PlasmaState.merge_parameters error ]
                \r>> The given 'parameter_values' dictionary must contain all
                \r>> parameter names as keys. The missing names are:
                \r>> {missing_params}
                """
            )

        for tag, slc in cls.slices.items():
            theta[slc] = parameter_values.get(tag)
        return theta

    @classmethod
    def get_values(
        cls, parameters: list[ParameterVector], field_requests: list[FieldRequest]
    ):
        param_values = {p.name: cls.theta[cls.slices[p.name]] for p in parameters}
        field_values = {
            f.name: cls.fields[f.name].get_values(cls.theta[cls.slices[f.name]], f)
            for f in field_requests
        }
        return param_values, field_values

    @classmethod
    def get_values_and_jacobians(
        cls, parameters: list[ParameterVector], field_requests: list[FieldRequest]
    ):
        param_values = {p.name: cls.theta[cls.slices[p.name]] for p in parameters}
        field_values = {}
        field_jacobians = {}
        for f in field_requests:
            field_model = cls.fields[f.name]
            field_params = cls.theta[cls.slices[f.name]]
            values, jacobian = field_model.get_values_and_jacobian(field_params, f)

            field_values[f.name] = values
            field_jacobians[f.name] = jacobian

        return param_values, field_values, field_jacobians
