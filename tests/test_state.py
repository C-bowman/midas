from numpy import array
from utilities import Polynomial, StraightLine
from midas.likelihoods import GaussianLikelihood, DiagnosticLikelihood
from midas import PlasmaState

def test_build_bounds():
    x, y, sigma = StraightLine.testing_data()

    poly_model = Polynomial(x_axis=x, order=2)
    likelihood = GaussianLikelihood(y_data=y, sigma=sigma)
    diagnostic = DiagnosticLikelihood(
        diagnostic_model=poly_model,
        likelihood=likelihood,
        name="poly"
    )

    PlasmaState.build_posterior(
        diagnostics=[diagnostic],
        priors=[],
        field_models=[]
    )

    # first test that we can assign all parameters the same bounds with a tuple
    param_bounds = {
        "poly_coefficients": (-10.0, 10.0),
    }
    bounds = PlasmaState.build_bounds(param_bounds)

    # now test we can assign different bounds using an array of the correct shape
    param_bounds = {
        "poly_coefficients": array([(-1, 1), (-2, 2), (-3, 3)]),
    }
    bounds = PlasmaState.build_bounds(param_bounds)
