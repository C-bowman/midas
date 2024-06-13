from numpy import ndarray, zeros
from midas import PlasmaState, FieldRequest, ParameterVector


class ForwardModel:
    parameters: list[ParameterVector]
    field_requests: list[FieldRequest]


class DiagnosticLikelihood:
    def __init__(self, forward_model: ForwardModel, likelihood: callable):
        self.forward_model = forward_model
        self.likelihood = likelihood
        self.field_requests = self.forward_model.field_requests
        self.parameters = self.forward_model.parameters

    def __call__(self) -> float:
        param_values, field_values = PlasmaState.get_values(
            parameters=self.parameters,
            field_requests=self.field_requests
        )

        predictions = self.forward_model.predictions(
            **param_values, **field_values
        )

        return self.likelihood._log_likelihood(predictions)

    def gradient(self) -> ndarray:
        param_values, field_values, field_jacobians = PlasmaState.get_values_and_jacobian(
            parameters=self.parameters,
            field_requests=self.field_requests
        )

        predictions, model_jacobians = self.forward_model.prediction_and_jacobian(
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