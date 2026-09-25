from numpy import atleast_1d, full, ndarray
from scipy.sparse import sparray

from midas.parameters import FieldRequest, Fields, Parameters, ParameterVector
from midas.posterior import BasePrior
from midas.validation import validate_name, validate_numeric_input


class LinearGaussianPrior(BasePrior):
	r"""
	Specify independent Gaussian priors on linear combinations of field values
	or parameters.

	If ``v`` is the target vector and ``A`` is the given operator, this class
	evaluates the unnormalised log-probability
	:math:`\log p(v) = -\frac{1}{2}\sum_i
	\left((A v)_i - \mu_i\right)^2 / \sigma_i^2`.

	This can be used to penalise profile gradients or curvature by supplying a
	finite-difference operator, or to constrain other linear combinations of a
	target vector.

	:param name:
		The name used to identify the prior.

	:param operator:
		A finite, real, two-dimensional ``numpy.ndarray`` or ``scipy.sparse``
		array with shape ``(m, n)``, where ``n`` is the number of
		target values. Sparse operators are stored without densifying them.

	:param mean:
		A finite, real ``float`` applied to every operator output, or a
		one-dimensional array with shape ``(m,)`` containing one Gaussian mean
		for each operator output.

	:param standard_deviation:
		A positive, finite ``float`` applied to every operator output, or a
		one-dimensional array with shape ``(m,)`` containing one positive
		Gaussian standard deviation for each operator output.

	:param field_request:
		A ``FieldRequest`` specifying the field values to which the prior is
		applied.

	:param parameter_vector:
		A ``ParameterVector`` specifying the parameters to which the prior is
		applied.
	"""

	def __init__(
		self,
		name: str,
		operator: ndarray | sparray,
		mean: ndarray | float,
		standard_deviation: ndarray | float,
		field_request: FieldRequest | None = None,
		parameter_vector: ParameterVector | None = None,
	):
		validate_name(name, error_source="LinearGaussianPrior")
		self.name = name

		if isinstance(field_request, FieldRequest):
			self.target = field_request.name
			self.target_type = "field_request"
			self.n_targets = field_request.size
			self.fields = Fields(field_request)
			self.parameters = Parameters()
			
		elif isinstance(parameter_vector, ParameterVector):
			self.target = parameter_vector.name
			self.target_type = "parameter_vector"
			self.n_targets = parameter_vector.size
			self.fields = Fields()
			self.parameters = Parameters(parameter_vector)
			
		else:
			raise ValueError(
				"LinearGaussianPrior requires either a 'field_request' or "
				"'parameter_vector'."
			)

		if not isinstance(operator, (ndarray, sparray)):
			raise TypeError(
				"LinearGaussianPrior 'operator' must be a numpy array or a "
				"scipy sparse array."
			)
		if operator.ndim != 2:
			raise ValueError("LinearGaussianPrior 'operator' must be two-dimensional.")
		if operator.shape[1] != self.n_targets:
			raise ValueError(
				"LinearGaussianPrior 'operator' must have one column for each "
				"target value."
			)
		
		validate_numeric_input(
			values=operator.tocoo().data if isinstance(operator, sparray) else operator,
			error_source="LinearGaussianPrior",
			input_name="operator",
		)
		self.A = operator

		output_shape = (operator.shape[0],)
		self.mean = atleast_1d(mean)
		self.mean = full(output_shape, self.mean) if self.mean.size == 1 else self.mean
		validate_numeric_input(
			values=self.mean,
			shape=output_shape,
			shape_name="operator output",
			error_source="LinearGaussianPrior",
			input_name="mean",
		)

		self.sigma = atleast_1d(standard_deviation)
		self.sigma = full(output_shape, self.sigma) if self.sigma.size == 1 else self.sigma
		validate_numeric_input(
			values=self.sigma,
			shape=output_shape,
			shape_name="operator output",
			error_source="LinearGaussianPrior",
			input_name="standard_deviation",
			limits=(0.0, float("inf")),
			strict_limits=True,
		)
		self.inv_sigma_sqr = 1.0 / self.sigma**2

	def probability(self, **kwargs: ndarray) -> float:
		residual = self.A @ kwargs[self.target] - self.mean
		return -0.5 * (residual**2 * self.inv_sigma_sqr).sum()

	def gradients(self, **kwargs: ndarray) -> dict[str, ndarray]:
		residual = self.A @ kwargs[self.target] - self.mean
		return {self.target: -self.A.T @ (residual * self.inv_sigma_sqr)}
