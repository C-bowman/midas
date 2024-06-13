from numpy import ndarray
from midas.fields import FieldModel
from midas.parameters import ParameterVector, FieldRequest


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
    fields: dict[str, FieldModel]

    @classmethod
    def specify_field_models(cls, fields: dict[str, FieldModel]):
        cls.fields = fields

    @classmethod
    def build_parametrisation(cls, components):
        # First check that the requested fields and the modelled fields match each other
        assert cls.fields is not None
        requested_fields = set()
        [[requested_fields.add(f) for f in c.field_requests] for c in components]
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
        slice_sizes = sorted([(f.name, f.n_params) for f in cls.fields.values()])


        parameter_sizes = {}
        for c in components:
            for p in c.parameter_checks:
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
            sorted([t for t in cls.parameter_checks.items()], key=lambda x: x[0])
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

    def split_parameters(self, theta):
        return {tag: theta[slc] for tag, slc in self.slices.items()}

    @classmethod
    def get_values(cls, parameters: list[ParameterVector], field_requests: list[FieldRequest]):
        param_values = {p.name: cls.theta[cls.slices[p.name]] for p in parameters}
        field_values = {
            f.name: cls.fields[f.name].get_values(cls.theta[cls.slices[f.name]], f)
            for f in field_requests
        }
        return param_values, field_values

    @classmethod
    def get_values_and_jacobian(cls, parameters: list[ParameterVector], field_requests: list[FieldRequest]):
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
