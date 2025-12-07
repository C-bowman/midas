from abc import ABC, abstractmethod
from collections.abc import Sequence
from numpy import array, ndarray, zeros
from midas.models.fields import FieldModel
from midas.models import DiagnosticModel
from midas.parameters import ParameterVector, Parameters, Fields
from midas.parameters import validate_parameters, validate_field_requests


class LikelihoodFunction(ABC):
    """
    An abstract base-class for likelihood function.
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
        self.forward_model = diagnostic_model
        self.likelihood = likelihood
        self.name = name
        self.fields = self.forward_model.fields
        self.model_parameters = self.forward_model.parameters
        self.likelihood_parameters = self.likelihood.parameters

    def log_probability(self) -> float:
        param_values, field_values = PlasmaState.get_values(
            parameters=self.model_parameters, fields=self.fields
        )

        predictions = self.forward_model.predictions(**param_values, **field_values)
        likelihood_param_values = PlasmaState.get_parameter_values(
            self.likelihood_parameters
        )
        return self.likelihood.log_likelihood(predictions, **likelihood_param_values)

    def log_probability_gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = (
            PlasmaState.get_values_and_jacobians(
                parameters=self.model_parameters, fields=self.fields
            )
        )

        predictions, model_jacobians = self.forward_model.predictions_and_jacobians(
            **param_values, **field_values
        )

        likelihood_param_values = PlasmaState.get_parameter_values(
            self.likelihood_parameters
        )
        dL_dp, likelihood_gradients = self.likelihood.derivatives(
            predictions, **likelihood_param_values
        )

        grad = zeros(PlasmaState.n_params)
        for param_name, likelihood_grad in likelihood_gradients.items():
            slc = PlasmaState.slices[param_name]
            grad[slc] = likelihood_grad

        for param_name in param_values.keys():
            slc = PlasmaState.slices[param_name]
            grad[slc] = dL_dp @ model_jacobians[param_name]

        for field_param, field_jacobian in field_jacobians.items():
            field_name = PlasmaState.field_parameter_map[field_param]
            slc = PlasmaState.slices[field_param]
            grad[slc] = (dL_dp @ model_jacobians[field_name]) @ field_jacobian

        return grad

    def get_predictions(self):
        param_values, field_values = PlasmaState.get_values(
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


class BasePrior(ABC):
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
            parameters=self.parameters, fields=self.fields
        )

        return self.probability(**param_values, **field_values)

    def log_probability_gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = (
            PlasmaState.get_values_and_jacobians(
                parameters=self.parameters, fields=self.fields
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
    field_models: dict[str, FieldModel] = {}
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
        cls.field_models = {f.name: f for f in field_models}
        # first gather all the fields that have been requested by the components
        requested_fields = set()
        [
            [requested_fields.add(f.name) for f in c.fields]
            for c in cls.components
        ]

        # If fields have been requested, but no field models have been specified,
        # tell the user how to specify them
        modelled_fields = {f for f in cls.field_models.keys()}
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
        for field_name, field_model in cls.field_models.items():
            cls.field_parameter_map.update(
                {param.name: field_name for param in field_model.parameters}
            )

        # Gather all the ParameterVector object in the analysis
        all_parameters = []
        [all_parameters.extend(d.model_parameters) for d in diagnostics]
        [all_parameters.extend(d.likelihood_parameters) for d in diagnostics]
        [all_parameters.extend(p.parameters) for p in priors]
        [all_parameters.extend(f.parameters) for f in field_models]

        # get the sizes of all unique ParameterVectors
        parameter_sizes = {}
        for p in all_parameters:
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
    def build_bounds(cls, parameter_bounds: dict[str, ndarray | tuple]) -> ndarray:
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
        bounds = zeros([cls.n_params, 2])

        missing_params = cls.parameter_names - {k for k in parameter_bounds.keys()}
        if len(missing_params) > 0:
            raise ValueError(
                f"""\n
                \r[ PlasmaState.build_bounds error ]
                \r>> The given 'parameter_bounds' dictionary must contain all
                \r>> parameter names as keys. The missing names are:
                \r>> {missing_params}
                """
            )

        for tag, slc in cls.slices.items():
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
                    \r[ PlasmaState.build_bounds error ]
                    \r>> The given bounds for each parameter must either be a 2D array
                    \r>> of shape ``(n_values, 2)``, where ``n_values`` is the number of
                    \r>> parameter values associated with a given parameter name, or
                    \r>> a 1D array with only two elements, in which case all values
                    \r>> will be assigned the same upper and lower bounds.
                    """
                )
        return bounds

    @classmethod
    def get_parameter_values(cls, parameters: Parameters):
        return {p.name: cls.theta[cls.slices[p.name]] for p in parameters}

    @classmethod
    def get_values(
        cls, parameters: Parameters, fields: Fields
    ):
        param_values = cls.get_parameter_values(parameters)
        field_values = {}
        for f in fields:
            field_model = cls.field_models[f.name]
            field_params = cls.get_parameter_values(field_model.parameters)
            field_values[f.name] = field_model.get_values(field_params, f)
        return param_values, field_values

    @classmethod
    def get_values_and_jacobians(
        cls, parameters: Parameters, fields: Fields
    ):
        param_values = cls.get_parameter_values(parameters)
        field_values = {}
        field_param_jacobians = {}
        for f in fields:
            field_model = cls.field_models[f.name]
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

            description = f"prior object at index {index} of the 'priors' argument"
            error_source = "PlasmaState.build_posterior"
            validate_parameters(prior, error_source, description)
            validate_field_requests(prior, error_source, description)

            if len(prior.parameters) == 0 and len(prior.fields) == 0:
                raise ValueError(
                    f"""
                    \r[ PlasmaState.build_posterior error ]
                    \r>> The prior object at index {index} of the 'priors' argument
                    \r>> has no specified field requests or parameters.
                    \r>>
                    \r>> At least one of the 'parameters' or 'fields' instance
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
