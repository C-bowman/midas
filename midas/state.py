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
    field_parameter_map: dict[str, str]
    components: list

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

        # loop over field models and add their parameters
        slice_sizes = []
        for field_model in cls.fields.values():
            slice_sizes.extend([(p.name, p.size) for p in field_model.parameters])
        # sort the field sizes by name
        slice_sizes = sorted(slice_sizes, key=lambda x: x[0])

        # now build a map between the names of parameter vectors of field models,
        # and the names of their parent fields:
        cls.field_parameter_map = {}
        for field_name, field_model in cls.fields.items():
            cls.field_parameter_map.update(
                {param.name: field_name for param in field_model.parameters}
            )

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
        cls.components = components

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
    def split_samples(cls, parameter_samples: ndarray) -> dict[str, ndarray]:
        valid_samples = (
            isinstance(parameter_samples, ndarray)
            and parameter_samples.ndim == 2
            and parameter_samples.shape[1] == cls.n_params
        )
        if not valid_samples:
            raise ValueError(
                f"""\n
                \r[ PlasmaState.split_samples error ]
                \r>> Given 'parameter_samples' argument must be an instance of a
                \r>> numpy.ndarray with shape (n, {cls.n_params}).
                """
            )
        return {tag: parameter_samples[:, slc] for tag, slc in cls.slices.items()}

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
    def get_parameter_values(cls, parameters: list[ParameterVector]):
        return {p.name: cls.theta[cls.slices[p.name]] for p in parameters}

    @classmethod
    def get_values(
        cls, parameters: list[ParameterVector], field_requests: list[FieldRequest]
    ):
        param_values = cls.get_parameter_values(parameters)
        field_values = {}
        for f in field_requests:
            field_model = cls.fields[f.name]
            field_params = cls.get_parameter_values(field_model.parameters)
            field_values[f.name] = field_model.get_values(field_params, f)
        return param_values, field_values

    @classmethod
    def get_values_and_jacobians(
        cls, parameters: list[ParameterVector], field_requests: list[FieldRequest]
    ):
        param_values = cls.get_parameter_values(parameters)
        field_values = {}
        field_param_jacobians = {}
        for f in field_requests:
            field_model = cls.fields[f.name]
            field_params = cls.get_parameter_values(field_model.parameters)
            values, jacobians = field_model.get_values_and_jacobian(field_params, f)

            field_values[f.name] = values
            field_param_jacobians.update(jacobians)

        return param_values, field_values, field_param_jacobians
