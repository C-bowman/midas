from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import chain
from numpy import ndarray, zeros
from midas.fields import FieldModel
from midas.models import DiagnosticModel
from midas.parameters import ParameterVector, FieldRequest


class LikelihoodFunction(ABC):
    """
    An abstract base-class for likelihood function.
    """

    @abstractmethod
    def log_likelihood(self, predictions: ndarray) -> float:
        """
        :param predictions: \
            The model predictions of the measured data as a 1D array.

        :return: \
            The calculated log-likelihood.
        """
        pass

    @abstractmethod
    def predictions_derivative(self, predictions: ndarray) -> ndarray:
        """
        :param predictions: \
            The model predictions of the measured data as a 1D array.

        :return: \
            The derivative of the log-likelihood with respect to each element of
            ``predictions`` as a 1D array.
        """
        pass


class DiagnosticLikelihood:
    """
    A class enabling the calculation of the likelihood (and its derivative) for the data
    of a particular diagnostic.

    :param diagnostic_model: \
        An instance of a diagnostic model which inherits from the ``DiagnosticModel``
        base class.

    :param likelihood: \
        An instance of a likelihood class which inherits from the ``LikelihoodFunction``
        base class.

    :param name: \
        A name or other identifier for the diagnostic as a string.

    """

    def __init__(
        self,
        diagnostic_model: DiagnosticModel,
        likelihood: LikelihoodFunction,
        name: str,
    ):
        self.forward_model = diagnostic_model
        self.likelihood = likelihood
        self.name = name
        self.field_requests = self.forward_model.field_requests
        self.parameters = self.forward_model.parameters

    def log_probability(self) -> float:
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        predictions = self.forward_model.predictions(**param_values, **field_values)

        return self.likelihood.log_likelihood(predictions)

    def log_probability_gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = (
            PlasmaState.get_values_and_jacobians(
                parameters=self.parameters, field_requests=self.field_requests
            )
        )

        predictions, model_jacobians = self.forward_model.predictions_and_jacobians(
            **param_values, **field_values
        )

        dL_dp = self.likelihood.predictions_derivative(predictions)

        grad = zeros(PlasmaState.n_params)
        for p in param_values.keys():
            slc = PlasmaState.slices[p]
            grad[slc] = dL_dp @ model_jacobians[p]

        for field_param in field_jacobians.keys():
            field_name = PlasmaState.field_parameter_map[field_param]
            slc = PlasmaState.slices[field_param]
            grad[slc] = (dL_dp @ model_jacobians[field_name]) @ field_jacobians[
                field_param
            ]

        return grad

    def get_predictions(self):
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        return self.forward_model.predictions(**param_values, **field_values)


class BasePrior(ABC):
    parameters: list[ParameterVector]
    field_requests: list[FieldRequest]
    name: str

    @abstractmethod
    def probability(self, **parameters_and_fields: ndarray) -> float:
        """
        Calculate the prior log-probability.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``field_requests``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The prior log-probability value.
        """
        pass

    @abstractmethod
    def gradients(self, **parameters_and_fields: ndarray) -> dict[str, ndarray]:
        """
        Calculate the prior log-probability.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``field_requests``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The gradient of the prior log-probability with respect to the given
            parameter and field values. These gradients are returned as a dictionary
            mapping the parameter and field names to their respective gradients as
            1D arrays.
        """

    def log_probability(self) -> float:
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        return self.probability(**param_values, **field_values)

    def log_probability_gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = (
            PlasmaState.get_values_and_jacobians(
                parameters=self.parameters, field_requests=self.field_requests
            )
        )

        gradients = self.gradients(**param_values, **field_values)

        grad = zeros(PlasmaState.n_params)
        for p in param_values.keys():
            slc = PlasmaState.slices[p]
            grad[slc] = gradients[p]

        for field_param in field_jacobians.keys():
            field_name = PlasmaState.field_parameter_map[field_param]
            slc = PlasmaState.slices[field_param]
            grad[slc] = gradients[field_name] @ field_jacobians[field_param]

        return grad


class PlasmaState:
    theta: ndarray
    radius: ndarray
    n_params: int
    parameter_names: set[str]
    parameter_sizes: dict[str, int]
    slices: dict[str, slice] = {}
    fields: dict[str, FieldModel] = {}
    field_parameter_map: dict[str, str]
    components: list[DiagnosticLikelihood | BasePrior]

    @classmethod
    def build_posterior(
        cls,
        diagnostics: list[DiagnosticLikelihood],
        priors: list[BasePrior],
        field_models: list[FieldModel],
    ):
        """
        Build the parametrisation for the posterior distribution by specifying the
        diagnostic likelihoods and prior distributions of which it is comprised,
        and models for any fields whose values are requested by those components.

        Each of the given components of the posterior are treated as independent, such
        that the posterior log-probability is given by the sum of the component
        log-probabilities.

        After this function has been called, the ``midas.posterior`` module can be used
        to evaluate the posterior log-probability and its gradient.

        :param diagnostics: \
            A ``list`` of ``DiagnosticLikelihood`` objects representing each diagnostic
            included in the analysis.

        :param priors: \
            A ``list`` containing instances of prior distribution classes which inherit
            from ``BasePrior`` representing the various components which make up the
            overall prior distribution.

        :param field_models: \
            A ``list`` of ``FieldModel`` objects, which represent all the fields
            being modelled in the analysis.
        """
        cls.__validate_diagnostics(diagnostics)
        cls.__validate_priors(priors)
        cls.__validate_field_models(field_models)

        cls.components = [*diagnostics, *priors]
        cls.fields = {f.name: f for f in field_models}
        # first gather all the fields that have been requested by the components
        requested_fields = set()
        [
            [requested_fields.add(f.name) for f in c.field_requests]
            for c in cls.components
        ]

        # If fields have been requested, but no field models have been specified,
        # tell the user how to specify them
        modelled_fields = {f for f in cls.fields.keys()}
        if len(modelled_fields) == 0 and len(requested_fields) > 0:
            raise ValueError(
                f"""\n
                \r[ PlasmaState.build_posterior error ]
                \r>> No models for the fields have been specified.
                \r>> Use 'PlasmaState.specify_field_models' to specify models
                \r>> for each of the requested fields in the analysis.
                \r>> The requested fields are:
                \r>> {requested_fields}
                """
            )

        # If field models have been specified, but they do not match the requested
        # fields, show the mismatch
        if modelled_fields != requested_fields:
            raise ValueError(
                f"""\n
                \r[ PlasmaState.build_posterior error ]
                \r>> The set of fields requested by the diagnostic likelihoods and / or
                \r>> priors does not match the set of modelled fields.
                \r>> The requested fields are:
                \r>> {requested_fields}
                \r>> but the modelled fields are:
                \r>> {modelled_fields}
                """
            )

        # Build a map between the names of parameter vectors of field models,
        # and the names of their parent fields:
        cls.field_parameter_map = {}
        for field_name, field_model in cls.fields.items():
            cls.field_parameter_map.update(
                {param.name: field_name for param in field_model.parameters}
            )

        # Collect the sizes of all unique ParameterVector objects in the analysis
        parameter_sizes = {}
        for f in chain(cls.components, cls.fields.values()):
            for p in f.parameters:
                assert isinstance(p, ParameterVector)
                if p.name not in parameter_sizes:
                    parameter_sizes[p.name] = p.size
                elif parameter_sizes[p.name] != p.size:
                    raise ValueError(
                        f"""\n
                        \r[ PlasmaState.build_posterior error ]
                        \r>> Two instances of 'ParameterVector' have matching names '{p.name}'
                        \r>> but differ in their size:
                        \r>> sizes are '{p.size}' and '{parameter_sizes[p.name]}'
                        """
                    )

        # sort the parameter sizes by name
        slice_sizes = sorted([t for t in parameter_sizes.items()], key=lambda x: x[0])
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
        cls.parameter_sizes = {name: s.stop - s.start for name, s in cls.slices.items()}

    @classmethod
    def split_parameters(cls, theta: ndarray) -> dict[str, ndarray]:
        """
        Split an array of all posterior parameters into sub-arrays corresponding to
        each named parameter set, and return a dictionary mapping the parameter set
        names to the associated sub-arrays.

        :param theta: \
            A full set of posterior parameter values as a 1D array.

        :return: \
            A dictionary mapping the names of parameter sub-sets to the corresponding
            sub-arrays of the posterior parameters.
        """
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
        """
        Split an array of posterior parameter samples into sub-arrays corresponding to
        samples of each named parameter set, and return a dictionary mapping the parameter
        set names to the associated sub-arrays.

        :param parameter_samples: \
            Samples from the posterior distribution as a 2D of shape
            ``(n_samples, n_parameters)``.

        :return: \
            A dictionary mapping the names of parameter sub-sets to the corresponding
            sub-arrays of the posterior samples.
        """
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
        """
        Merge the values of named parameter sub-sets into a single array of posterior
        parameter values.

        :param parameter_values: \
            A dictionary mapping the names of parameter sub-sets to arrays of values
            for those parameters.

        :return: \
            A 1D array of posterior parameter values.
        """
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

    @staticmethod
    def __validate_diagnostics(diagnostics: Sequence):
        if not isinstance(diagnostics, Sequence):
            raise TypeError(
                f"""\n
                \r[ PlasmaState.build_posterior error ]
                \r>> The 'diagnostics' argument must be a sequence,
                \r>> but instead has type
                \r>> {type(diagnostics)}
                \r>> which is not a sequence.
                """
            )

        for index, diagnostic in enumerate(diagnostics):
            if not isinstance(diagnostic, DiagnosticLikelihood):
                raise TypeError(
                    f"""\n
                    \r[ PlasmaState.build_posterior error ]
                    \r>> The 'diagnostics' argument must contain only instances
                    \r>> ``DiagnosticLikelihood``, but the object at index {index}
                    \r>> instead has type:
                    \r>> {type(diagnostic)}
                    """
                )

    @staticmethod
    def __validate_priors(priors: Sequence):
        if not isinstance(priors, Sequence):
            raise TypeError(
                f"""\n
                \r[ PlasmaState.build_posterior error ]
                \r>> The 'priors' argument must be a sequence,
                \r>> but instead has type
                \r>> {type(priors)}
                \r>> which is not a sequence.
                """
            )

        for index, prior in enumerate(priors):
            if not isinstance(prior, BasePrior):
                raise TypeError(
                    f"""\n
                    \r[ PlasmaState.build_posterior error ]
                    \r>> The 'priors' argument must contain only instances of
                    \r>> classes which inherit from ``BasePrior``, but the object 
                    \r>> at index {index} instead has type:
                    \r>> {type(prior)}
                    """
                )

            valid_parameters = (
                hasattr(prior, "parameters")
                and isinstance(prior.parameters, Sequence)
                and all(isinstance(p, ParameterVector) for p in prior.parameters)
            )
            if not valid_parameters:
                raise TypeError(
                    f"""\n
                    \r[ PlasmaState.build_posterior error ]
                    \r>> The prior object at index {index} of the 'priors' argument
                    \r>> does not possess a valid 'parameters' instance attribute.
                    \r>>
                    \r>> 'parameters' must be a list containing only instances of the
                    \r>> ``ParameterVector`` class (or an empty list).
                    """
                )

            # check that all the parameter names in the current prior are unique
            parameter_names = set()
            for p in prior.parameters:
                if p.name not in parameter_names:
                    parameter_names.add(p.name)
                else:
                    raise TypeError(
                        f"""\n
                        \r[ PlasmaState.build_posterior error ]
                        \r>> The prior object at index {index} of the 'priors' argument
                        \r>> does not possess a valid 'parameters' instance attribute.
                        \r>>
                        \r>> At least two ``ParameterVector`` objects share the name:
                        \r>> '{p.name}'
                        \r>> but all names must be unique.
                        """
                    )

            valid_field_requests = (
                hasattr(prior, "field_requests")
                and isinstance(prior.field_requests, Sequence)
                and all(isinstance(f, FieldRequest) for f in prior.field_requests)
            )
            if not valid_field_requests:
                raise TypeError(
                    f"""\n
                    \r[ PlasmaState.build_posterior error ]
                    \r>> The prior object at index {index} of the 'priors' argument
                    \r>> does not possess a valid 'field_requests' instance attribute.
                    \r>>
                    \r>> 'field_requests' must be a list containing only instances of
                    \r>> the ``FieldRequest`` class (or an empty list).
                    """
                )

            # check that all the requested fields in the current prior are unique
            field_names = set()
            for f in prior.field_requests:
                if f.name not in field_names:
                    field_names.add(f.name)
                else:
                    raise TypeError(
                        f"""\n
                        \r[ PlasmaState.build_posterior error ]
                        \r>> The prior object at index {index} of the 'priors' argument
                        \r>> does not possess a valid 'field_requests' instance attribute.
                        \r>>
                        \r>> At least two ``FieldRequest`` objects request the same field:
                        \r>> '{f.name}'
                        \r>> but all ``FieldRequest`` objects must request unique fields.
                        """
                    )

            if len(prior.parameters) == 0 and len(prior.field_requests) == 0:
                raise ValueError(
                    f"""
                    \r[ PlasmaState.build_posterior error ]
                    \r>> The prior object at index {index} of the 'priors' argument
                    \r>> has no specified field requests or parameters.
                    \r>>
                    \r>> At least one of the 'parameters' or 'field_requests' instance
                    \r>> attributes must be non-empty.
                    """
                )

    @staticmethod
    def __validate_field_models(field_models: list[FieldModel]):
        # first check that the given models are valid:
        valid_models = isinstance(field_models, Sequence) and all(
            isinstance(model, FieldModel) for model in field_models
        )
        if not valid_models:
            raise ValueError(
                """
                \r[ PlasmaState.build_posterior error ]
                \r>> Given 'field_models' must be a sequence of objects
                \r>> whose types derive from the 'FieldModel' abstract base class.
                """
            )

        # check that each model is for a unique field
        unique_fields = len({f.name for f in field_models}) == len(field_models)
        if not unique_fields:
            raise ValueError(
                """
                \r[ PlasmaState.build_posterior error ]
                \r>> The given field models must each specify a unique field name.
                """
            )