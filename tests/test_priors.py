from midas.priors import GaussianProcessPrior
from midas.fields import PiecewiseLinearField, FieldRequest
from midas.state import PlasmaState


from numpy import linspace, sin
from numpy.random import default_rng
from scipy.optimize import approx_fprime


def test_gp_prior():
    # build a linear field
    R = linspace(1, 10, 10)
    linear_field = PiecewiseLinearField(
        field_name="emission",
        axis_name="radius",
        axis=R
    )

    # generate some random positions at which to request field values
    rng = default_rng(2391)
    random_positions = rng.normal(loc=1, scale=0.15, size=16).cumsum()
    random_positions *= (10 / random_positions[-1])
    request = FieldRequest(name="emission", coordinates={"radius": random_positions})

    # set up a posterior containing only a gaussian process prior
    gp_prior = GaussianProcessPrior(
        name="emission",
        field_positions=request,
    )

    PlasmaState.specify_field_models([linear_field])
    PlasmaState.build_parametrisation([gp_prior])

    # build some test parameters at which to evaluate the posterior
    param_dict = {
        "emission_linear_basis": sin(0.5 * R),
        "emission_mean_hyperpars": [0.05],
        "emission_cov_hyperpars": [1.0, -1.1]
    }
    param_array = PlasmaState.merge_parameters(param_dict)

    # evaluate the posterior gradient both analytically and numerically
    from midas import posterior
    analytic_grad = posterior.gradient(param_array)
    numeric_grad = approx_fprime(xk=param_array, f=posterior.log_probability)

    # check that the fractional error between the gradients is small
    frac_err = numeric_grad / analytic_grad - 1
    assert abs(frac_err).max() < 1e-4
