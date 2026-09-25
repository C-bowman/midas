from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from numpy import array, ndarray, zeros
from midas.models.fields import FieldModel
from midas.models import DiagnosticModel
from midas.parameters import FieldRequest, ParameterVector, Parameters, Fields
from midas.parameters import validate_parameters, validate_field_requests


class _EvaluationContext:
    """Resolve parameter and field values for one posterior evaluation."""

    def __init__(self, state: "Posterior", theta: ndarray):
        self.state = state
        self.theta = theta.copy()
        self.parameter_values = state.split_parameters(self.theta)
        self._field_values = {}
        self._field_jacobians = {}

    @property
    def n_params(self) -> int:
        return self.state.n_params

    @property
    def slices(self) -> Mapping[str, slice]:
        return self.state.slices

    def get_parameter_values(self, parameters: Parameters) -> dict[str, ndarray]:
        return {p.name: self.parameter_values[p.name] for p in parameters}

    def get_field_values(self, field):
        if field not in self._field_values:
            field_model = self.state.field_models[field.name]
            field_params = self.get_parameter_values(field_model.parameters)
            self._field_values[field] = field_model.get_values(field_params, field)
        return self._field_values[field]

    def get_field_values_and_jacobians(self, field):
        if field not in self._field_jacobians:
            field_model = self.state.field_models[field.name]
            field_params = self.get_parameter_values(field_model.parameters)
            values, jacobians = field_model.get_values_and_jacobian(
                field_params, field
            )
            self._field_values[field] = values
            self._field_jacobians[field] = jacobians
        return self._field_values[field], self._field_jacobians[field]

    def get_values(self, parameters: Parameters, fields: Fields):
        param_values = self.get_parameter_values(parameters)
        field_values = {f.name: self.get_field_values(f) for f in fields}
        return param_values, field_values

    def get_values_and_jacobians(
        self, parameters: Parameters, fields: Fields
    ) -> tuple[dict[str, ndarray], dict[str, ndarray], dict[str, dict[str, ndarray]]]:
        param_values = self.get_parameter_values(parameters)
        field_values = {}
        field_jacobians = {}
        for field in fields:
            values, jacobians = self.get_field_values_and_jacobians(field)
            field_values[field.name] = values
            field_jacobians[field.name] = jacobians
        return param_values, field_values, field_jacobians


class LikelihoodFunction(ABC):
    """
    An abstract base class for likelihood functions.
    """
    parameters: Parameters

    @abstractmethod
    def log_likelihood(self, predictions: ndarray, **parameters: ndarray) -> float:
        """
        :param predictions: \
            The model predictions of the measured data as a 1D array.

        :return: \
            The calculated log-likelihood.
        """
        pass

    @abstractmethod
    def derivatives(
        self, predictions: ndarray, **parameters: ndarray
    ) -> ndarray:
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
        self.__validate_diagnostic_model(diagnostic_model)
        self.__validate_likelihood(likelihood)
        self.forward_model = diagnostic_model
        self.likelihood = likelihood
        self.name = name
        self.fields = self.forward_model.fields
        self.model_parameters = self.forward_model.parameters
        self.likelihood_parameters = self.likelihood.parameters

    def log_probability(self, context: _EvaluationContext) -> float:
        param_values, field_values = context.get_values(
            parameters=self.model_parameters, fields=self.fields
        )

        predictions = self.forward_model.predictions(**param_values, **field_values)
        likelihood_param_values = context.get_parameter_values(
            self.likelihood_parameters
        )
        return self.likelihood.log_likelihood(predictions, **likelihood_param_values)

    def log_probability_gradient(self, context: _EvaluationContext) -> ndarray:
        param_values, field_values, field_jacobians = (
            context.get_values_and_jacobians(
                parameters=self.model_parameters, fields=self.fields
            )
        )

        predictions, model_jacobians = self.forward_model.predictions_and_jacobians(
            **param_values, **field_values
        )

        likelihood_param_values = context.get_parameter_values(
            self.likelihood_parameters
        )
        dL_dp, likelihood_gradients = self.likelihood.derivatives(
            predictions, **likelihood_param_values
        )

        grad = zeros(context.n_params)
        for param_name, likelihood_grad in likelihood_gradients.items():
            slc = context.slices[param_name]
            grad[slc] += likelihood_grad

        for param_name in param_values.keys():
            slc = context.slices[param_name]
            grad[slc] += dL_dp @ model_jacobians[param_name]

        for field_name, jacobians in field_jacobians.items():
            field_gradient = dL_dp @ model_jacobians[field_name]
            for param_name, jacobian in jacobians.items():
                slc = context.slices[param_name]
                grad[slc] += field_gradient @ jacobian

        return grad

    def get_predictions(self, context: _EvaluationContext):
        param_values, field_values = context.get_values(
            parameters=self.model_parameters, fields=self.fields
        )

        return self.forward_model.predictions(**param_values, **field_values)

    @staticmethod
    def __validate_diagnostic_model(diagnostic_model: DiagnosticModel):

        if not isinstance(diagnostic_model, DiagnosticModel):
            raise TypeError(
                f"""\n
                \r[ DiagnosticLikelihood error ]
                \r>> The 'diagnostic_model' argument must be an instance of
                \r>> ``DiagnosticModel``, but instead has type:
                \r>> {type(diagnostic_model)}
                """
            )

        error_source = "DiagnosticLikelihood"
        description = "given 'diagnostic_model'"
        validate_parameters(diagnostic_model, error_source, description)
        validate_field_requests(diagnostic_model, error_source, description)

    @staticmethod
    def __validate_likelihood(likelihood: LikelihoodFunction):
        if not isinstance(likelihood, LikelihoodFunction):
            raise TypeError(
                f"""\n
                \r[ DiagnosticLikelihood error ]
                \r>> The 'likelihood' argument must be an instance of
                \r>> ``LikelihoodFunction``, but instead has type:
                \r>> {type(likelihood)}
                """
            )

        validate_parameters(
            likelihood,
            error_source="DiagnosticLikelihood",
            description="given 'likelihood'",
        )


class BasePrior(ABC):
    """
    An abstract base class for prior probability distributions.

    Subclasses define the log-probability and its gradients for the parameter vectors
    and field values listed in their ``parameters`` and ``fields`` attributes.
    """

    parameters: Parameters
    fields: Fields
    name: str

    @abstractmethod
    def probability(self, **parameters_and_fields: ndarray) -> float:
        """
        Calculate the prior log-probability.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``fields``
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
        Calculate the gradients of the prior log-probability.

        :param parameters_and_fields: \
            The parameter and field values requested via the ``ParameterVector`` and
            ``FieldRequest`` objects stored in ``parameters`` and ``fields``
            instance variables.

            The names of the unpacked keyword arguments correspond to the ``name``
            attribute of the ``ParameterVector`` and ``FieldRequest`` objects, and
            their values will be passed as 1D arrays.

        :return: \
            The gradient of the prior log-probability with respect to the given
            parameter and field values. These gradients are returned as a dictionary
            mapping the parameter and field names to their respective gradients as
            1D arrays.

            These must be partial derivatives with all other inputs held fixed.
            Contributions through fields that depend on a requested parameter are
            propagated separately and added to its direct gradient.
        """

    def log_probability(self, context: _EvaluationContext) -> float:
        param_values, field_values = context.get_values(
            parameters=self.parameters, fields=self.fields
        )

        return self.probability(**param_values, **field_values)

    def log_probability_gradient(self, context: _EvaluationContext) -> ndarray:
        param_values, field_values, field_jacobians = (
            context.get_values_and_jacobians(
                parameters=self.parameters, fields=self.fields
            )
        )

        gradients = self.gradients(**param_values, **field_values)

        grad = zeros(context.n_params)
        for p in param_values.keys():
            slc = context.slices[p]
            grad[slc] += gradients[p]

        for field_name, jacobians in field_jacobians.items():
            for param_name, jacobian in jacobians.items():
                slc = context.slices[param_name]
                grad[slc] += gradients[field_name] @ jacobian

        return grad


class Posterior:
    """
    A validated, self-contained MIDAS posterior distribution.

    Construction validates the posterior components and creates the mappings between
    the flat posterior parameter vector, named parameter vectors, and field models.
    Parameter values used during evaluation are held in a short-lived context rather
    than on the posterior instance.
    """

    n_params: int
    parameter_names: tuple[str, ...]
    parameter_set: frozenset[str]
    parameter_sizes: Mapping[str, int]
    slices: Mapping[str, slice]
    field_models: Mapping[str, FieldModel]
    components: tuple[DiagnosticLikelihood | BasePrior, ...]

    def __init__(
        self,
        diagnostics: Sequence[DiagnosticLikelihood],
        priors: Sequence[BasePrior],
        field_models: Sequence[FieldModel],
    ):
        """
        Build the parametrisation for the posterior distribution by specifying the
        diagnostic likelihoods and prior distributions of which it is comprised,
        and models for any fields whose values are requested by those components.

        Each of the given components of the posterior are treated as independent, such
        that the posterior log-probability is given by the sum of the component
        log-probabilities.

        :param diagnostics: \
            A sequence of ``DiagnosticLikelihood`` objects representing each
            diagnostic included in the analysis.

        :param priors: \
            A sequence containing instances of prior distribution classes which
            inherit from ``BasePrior`` representing the components of the prior.

        :param field_models: \
            A sequence of ``FieldModel`` objects, which represent all the fields
            being modelled in the analysis.
        """
        self.__validate_diagnostics(diagnostics)
        self.__validate_priors(priors)
        self.__validate_field_models(field_models)
        self.__validate_component_names([*diagnostics, *priors])

        self.components = (*diagnostics, *priors)
        self.field_models = MappingProxyType({f.name: f for f in field_models})
        # first gather all the fields that have been requested by the components
        requested_fields = set()
        [
            [requested_fields.add(f.name) for f in c.fields]
            for c in self.components
        ]

        # If fields have been requested, but no field models have been specified,
        # tell the user how to specify them
        modelled_fields = set(self.field_models)
        if len(modelled_fields) == 0 and len(requested_fields) > 0:
            raise ValueError(
                f"""\n
                \r[ build_posterior error ]
                \r>> No models for the fields have been specified.
                \r>> The requested fields are:
                \r>> {requested_fields}
                """
            )

        # If field models have been specified, but they do not match the requested
        # fields, show the mismatch
        if modelled_fields != requested_fields:
            raise ValueError(
                f"""\n
                \r[ build_posterior error ]
                \r>> The set of fields requested by the diagnostic likelihoods and / or
                \r>> priors does not match the set of modelled fields.
                \r>> The requested fields are:
                \r>> {requested_fields}
                \r>> but the modelled fields are:
                \r>> {modelled_fields}
                """
            )

        # Gather all the ParameterVector object in the analysis
        all_parameters = []
        [all_parameters.extend(d.model_parameters) for d in diagnostics]
        [all_parameters.extend(d.likelihood_parameters) for d in diagnostics]
        [all_parameters.extend(p.parameters) for p in priors]
        [all_parameters.extend(f.parameters) for f in field_models]

        if len(all_parameters) == 0:
            raise ValueError(
                """
                \r[ build_posterior error ]
                \r>> The posterior must contain at least one parameter, but no
                \r>> parameters were specified by its diagnostics, likelihoods,
                \r>> priors or field models.
                """
            )

        # get the sizes of all unique ParameterVectors
        parameter_sizes = {}
        for p in all_parameters:
            assert isinstance(p, ParameterVector)
            if p.name not in parameter_sizes:
                parameter_sizes[p.name] = p.size
            elif parameter_sizes[p.name] != p.size:
                raise ValueError(
                    f"""\n
                    \r[ build_posterior error ]
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
        self.n_params = slices[-1][1].stop
        # convert to a dictionary which maps parameter names to corresponding
        # slices of the parameter vector
        slice_map = dict(slices)
        self.slices = MappingProxyType(slice_map)
        self.parameter_set = frozenset(slice_map)
        self.parameter_sizes = MappingProxyType({
            name: slc.stop - slc.start for name, slc in slice_map.items()
        })
        self.parameter_names = tuple(slice_map)
        self._components_by_name = MappingProxyType({
            component.name: component for component in self.components
        })

    def split_parameters(self, theta: ndarray) -> dict[str, ndarray]:
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
        if not isinstance(theta, ndarray) or theta.shape != (self.n_params,):
            raise ValueError(
                f"""\n
                \r[ Posterior.split_parameters error ]
                \r>> Given 'theta' argument must be an instance of a
                \r>> numpy.ndarray with shape ({self.n_params},).
                """
            )
        return {tag: theta[slc] for tag, slc in self.slices.items()}

    def split_samples(self, parameter_samples: ndarray) -> dict[str, ndarray]:
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
            and parameter_samples.shape[1] == self.n_params
        )
        if not valid_samples:
            raise ValueError(
                f"""\n
                \r[ Posterior.split_samples error ]
                \r>> Given 'parameter_samples' argument must be an instance of a
                \r>> numpy.ndarray with shape (n, {self.n_params}).
                """
            )
        return {tag: parameter_samples[:, slc] for tag, slc in self.slices.items()}

    def merge_parameters(self, parameter_values: dict[str, ndarray | float]) -> ndarray:
        """
        Merge the values of named parameter sub-sets into a single array of posterior
        parameter values.

        :param parameter_values: \
            A dictionary mapping the names of parameter sub-sets to arrays of values
            for those parameters.

        :return: \
            A 1D array of posterior parameter values.
        """
        theta = zeros(self.n_params)

        missing_params = self.parameter_set - set(parameter_values)
        if len(missing_params) > 0:
            raise ValueError(
                f"""\n
                \r[ Posterior.merge_parameters error ]
                \r>> The given 'parameter_values' dictionary must contain all
                \r>> parameter names as keys. The missing names are:
                \r>> {missing_params}
                """
            )

        for tag, slc in self.slices.items():
            theta[slc] = parameter_values.get(tag)
        return theta

    def build_bounds(self, parameter_bounds: dict[str, ndarray | tuple]) -> ndarray:
        """
        Given a dictionary mapping parameter vector names to arrays specifying the lower
        and upper bounds for those parameters, merge these bounds into a single 2D
        numpy array of shape ``(n_parameters, 2)``.

        :param parameter_bounds: \
            A dictionary mapping the names of parameter vectors to arrays specifying
            the lower and upper bounds for those parameters. The given bounds for each
            parameter must either be a 2D array of shape ``(n_values, 2)``, where
            ``n_values`` is the number of parameter values associated with a given
            parameter name, or a 1D array with only two elements, in which case all
            values will be assigned the same upper and lower bounds.

        :return: \
            The posterior parameter bounds as a 2D array.
        """
        bounds = zeros([self.n_params, 2])

        missing_params = self.parameter_set - set(parameter_bounds)
        if len(missing_params) > 0:
            raise ValueError(
                f"""\n
                \r[ Posterior.build_bounds error ]
                \r>> The given 'parameter_bounds' dictionary must contain all
                \r>> parameter names as keys. The missing names are:
                \r>> {missing_params}
                """
            )

        for tag, slc in self.slices.items():
            b = parameter_bounds.get(tag)
            b = b if isinstance(b, ndarray) else array(b)
            b = b.squeeze()
            if b.size == 2:
                bounds[slc, 0] = b[0]
                bounds[slc, 1] = b[1]
            elif b.shape == (slc.stop - slc.start, 2):
                bounds[slc, :] = b
            else:
                raise ValueError(
                    f"""\n
                    \r[ Posterior.build_bounds error ]
                    \r>> The given bounds for each parameter must either be a 2D array
                    \r>> of shape ``(n_values, 2)``, where ``n_values`` is the number of
                    \r>> parameter values associated with a given parameter name, or
                    \r>> a 1D array with only two elements, in which case all values
                    \r>> will be assigned the same upper and lower bounds.
                    """
                )
        return bounds

    def _context(self, theta: ndarray) -> _EvaluationContext:
        return _EvaluationContext(self, theta)

    def log_probability(self, theta: ndarray) -> float:
        context = self._context(theta)
        return sum(
            component.log_probability(context) for component in self.components
        )

    def gradient(self, theta: ndarray) -> ndarray:
        context = self._context(theta)
        return sum(
            component.log_probability_gradient(context)
            for component in self.components
        )

    def cost(self, theta: ndarray) -> float:
        return -self.log_probability(theta)

    def cost_gradient(self, theta: ndarray) -> ndarray:
        return -self.gradient(theta)

    def component_log_probabilities(self, theta: ndarray) -> dict[str, float]:
        context = self._context(theta)
        return {
            component.name: component.log_probability(context)
            for component in self.components
        }

    def component_log_probability(
        self, theta: ndarray, component_name: str
    ) -> float:
        context = self._context(theta)
        return self._components_by_name[component_name].log_probability(context)

    def component_gradient(
        self, theta: ndarray, component_name: str
    ) -> ndarray:
        context = self._context(theta)
        return self._components_by_name[component_name].log_probability_gradient(
            context
        )

    def get_model_predictions(self, theta: ndarray) -> dict[str, ndarray]:
        context = self._context(theta)
        return {
            component.name: component.get_predictions(context)
            for component in self.components
            if isinstance(component, DiagnosticLikelihood)
        }

    def sample_model_predictions(
        self, parameter_samples: ndarray
    ) -> dict[str, ndarray]:
        self.split_samples(parameter_samples)
        predictions = defaultdict(list)
        diagnostics = [
            component for component in self.components
            if isinstance(component, DiagnosticLikelihood)
        ]
        for theta in parameter_samples:
            context = self._context(theta)
            for diagnostic in diagnostics:
                predictions[diagnostic.name].append(
                    diagnostic.get_predictions(context)
                )
        return {name: array(values) for name, values in predictions.items()}

    def sample_field_values(
        self, parameter_samples: ndarray, field_request: FieldRequest
    ) -> ndarray:
        self.split_samples(parameter_samples)
        if field_request.name not in self.field_models:
            raise ValueError(
                f"No model was configured for field '{field_request.name}'."
            )

        field_model = self.field_models[field_request.name]
        field_values = zeros([parameter_samples.shape[0], field_request.size])
        for index, theta in enumerate(parameter_samples):
            context = self._context(theta)
            field_values[index, :] = field_model.get_values(
                context.get_parameter_values(field_model.parameters), field_request
            )
        return field_values

    @staticmethod
    def __validate_diagnostics(diagnostics: Sequence):
        if not isinstance(diagnostics, Sequence):
            raise TypeError(
                f"""\n
                \r[ build_posterior error ]
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
                    \r[ build_posterior error ]
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
                \r[ build_posterior error ]
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
                    \r[ build_posterior error ]
                    \r>> The 'priors' argument must contain only instances of
                    \r>> classes which inherit from ``BasePrior``, but the object
                    \r>> at index {index} instead has type:
                    \r>> {type(prior)}
                    """
                )

            description = f"prior object at index {index} of the 'priors' argument"
            error_source = "build_posterior"
            validate_parameters(prior, error_source, description)
            validate_field_requests(prior, error_source, description)

            if len(prior.parameters) == 0 and len(prior.fields) == 0:
                raise ValueError(
                    f"""
                    \r[ build_posterior error ]
                    \r>> The prior object at index {index} of the 'priors' argument
                    \r>> has no specified field requests or parameters.
                    \r>>
                    \r>> At least one of the 'parameters' or 'fields' instance
                    \r>> attributes must be non-empty.
                    """
                )

    @staticmethod
    def __validate_component_names(
        components: Sequence[DiagnosticLikelihood | BasePrior],
    ):
        component_names = []
        for index, component in enumerate(components):
            name = getattr(component, "name", None)
            if not isinstance(name, str) or len(name) == 0:
                raise ValueError(
                    f"""\n
                    \r[ build_posterior error ]
                    \r>> Every posterior component must have a non-empty string 'name'
                    \r>> attribute, but the component at index {index} has the name:
                    \r>> {name!r}
                    """
                )
            component_names.append(name)

        duplicate_names = sorted(
            name for name in set(component_names) if component_names.count(name) > 1
        )
        if duplicate_names:
            raise ValueError(
                f"""\n
                \r[ build_posterior error ]
                \r>> Every posterior component must have a unique name, but the
                \r>> following names are used by more than one component:
                \r>> {duplicate_names}
                """
            )

    @staticmethod
    def __validate_field_models(field_models: Sequence[FieldModel]):
        # first check that the given models are valid:
        valid_models = isinstance(field_models, Sequence) and all(
            isinstance(model, FieldModel) for model in field_models
        )
        if not valid_models:
            raise ValueError(
                """
                \r[ build_posterior error ]
                \r>> Given 'field_models' must be a sequence of objects
                \r>> whose types derive from the 'FieldModel' abstract base class.
                """
            )

        for index, model in enumerate(field_models):
            if not isinstance(model.name, str) or len(model.name) == 0:
                raise ValueError(
                    f"""\n
                    \r[ build_posterior error ]
                    \r>> Every field model must have a non-empty string 'name'
                    \r>> attribute, but the model at index {index} has the name:
                    \r>> {model.name!r}
                    """
                )

            validate_parameters(
                model,
                error_source="build_posterior",
                description=f"field model at index {index}",
            )

        # check that each model is for a unique field
        unique_fields = len({f.name for f in field_models}) == len(field_models)
        if not unique_fields:
            raise ValueError(
                """
                \r[ build_posterior error ]
                \r>> The given field models must each specify a unique field name.
                """
            )


def build_posterior(
    diagnostics: Sequence[DiagnosticLikelihood],
    priors: Sequence[BasePrior],
    field_models: Sequence[FieldModel],
) -> Posterior:
    """Validate posterior components and return an independent posterior."""
    return Posterior(diagnostics, priors, field_models)
