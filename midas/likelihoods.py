from numpy import ndarray, log, pi, zeros
from midas.state import PlasmaState
from midas.models import DiagnosticModel


class DiagnosticLikelihood:
    def __init__(self, diagnostic_model: DiagnosticModel, likelihood: callable):
        self.forward_model = diagnostic_model
        self.likelihood = likelihood
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

        for f in field_values.keys():
            slc = PlasmaState.slices[f]
            grad[slc] = (dL_dp @ model_jacobians[f]) @ field_jacobians[f]

        return grad

    def get_predictions(self):
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters, field_requests=self.field_requests
        )

        return self.forward_model.predictions(**param_values, **field_values)


class GaussianLikelihood:
    """
    A class for constructing a Gaussian likelihood function.

    :param y_data: \
        The measured data as a 1D array.

    :param sigma: \
        The standard deviations corresponding to each element in ``y_data`` as a 1D array.
    """

    def __init__(
        self,
        y_data: ndarray,
        sigma: ndarray,
    ):
        self.y = y_data
        self.sigma = sigma
        self.n_data = self.y.size
        self.inv_sigma = 1.0 / self.sigma
        self.inv_sigma_sqr = self.inv_sigma**2
        self.normalisation = -log(self.sigma).sum() - 0.5 * log(2 * pi) * self.n_data

    def log_likelihood(self, predictions):
        z = (self.y - predictions) * self.inv_sigma
        return -0.5 * (z**2).sum() + self.normalisation

    def predictions_derivative(self, predictions):
        return (self.y - predictions) * self.inv_sigma_sqr
