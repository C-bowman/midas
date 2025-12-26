import pytest
from numpy import linspace, sin
from scipy.optimize import approx_fprime
from numpy.random import default_rng

from midas.priors import GaussianProcessPrior, GaussianPrior, ExponentialPrior
from midas.models.fields import PiecewiseLinearField, FieldRequest
from midas.state import PlasmaState
from midas import posterior

rng = default_rng(2391)


def test_gp_prior():
    # build a linear field
    R = linspace(1, 10, 10)
    linear_field = PiecewiseLinearField(
        field_name="emission", axis_name="radius", axis=R
    )

    # generate some random positions at which to request field values
    random_positions = rng.normal(loc=1, scale=0.15, size=16).cumsum()
    random_positions *= 10 / random_positions[-1]
    request = FieldRequest(name="emission", coordinates={"radius": random_positions})

    # set up a posterior containing only a gaussian process prior
    gp_prior = GaussianProcessPrior(
        name="emission",
        field_positions=request,
    )

    PlasmaState.build_posterior(
        diagnostics=[], priors=[gp_prior], field_models=[linear_field]
    )

    # build some test parameters at which to evaluate the posterior
    param_dict = {
        "emission_linear_basis": sin(0.5 * R),
        "emission_mean_hyperpars": [0.05],
        "emission_cov_hyperpars": [1.0, -1.1],
    }
    param_array = PlasmaState.merge_parameters(param_dict)

    # evaluate the posterior gradient both analytically and numerically
    analytic_grad = posterior.gradient(param_array)
    numeric_grad = approx_fprime(xk=param_array, f=posterior.log_probability)

    # check that the fractional error between the gradients is small
    frac_err = numeric_grad / analytic_grad - 1
    assert abs(frac_err).max() < 1e-4

    # repeat the gradient calculation check after fixing the hyperparameters
    gp_prior.fix_hyperparameters(param_dict)
    PlasmaState.build_posterior(
        diagnostics=[], priors=[gp_prior], field_models=[linear_field]
    )

    param_array = PlasmaState.merge_parameters(param_dict)
    # evaluate the posterior gradient both analytically and numerically
    analytic_grad = posterior.gradient(param_array)
    numeric_grad = approx_fprime(xk=param_array, f=posterior.log_probability)
    # check that the fractional error between the gradients is small
    frac_err = numeric_grad / analytic_grad - 1
    assert abs(frac_err).max() < 1e-4


prior_test_setup = [
    (
        GaussianPrior,
        {
            "mean": rng.uniform(low=-1.0, high=1.0, size=16),
            "standard_deviation": rng.uniform(low=0.5, high=2.0, size=16),
        },
    ),
    (
        ExponentialPrior,
        {
            "mean": rng.uniform(low=0.1, high=10.0, size=16),
        },
    )
]


@pytest.mark.parametrize("prior_class, kwargs", prior_test_setup)
def test_unparameterized_priors(prior_class, kwargs):
    # build a linear field
    R = linspace(1, 10, 10)
    linear_field = PiecewiseLinearField(
        field_name="emission", axis_name="radius", axis=R
    )

    # generate some random positions at which to request field values
    random_positions = rng.normal(loc=1, scale=0.15, size=16).cumsum()
    random_positions *= 10 / random_positions[-1]
    request = FieldRequest(name="emission", coordinates={"radius": random_positions})

    prior = prior_class(
        name="emission",
        field_positions=request,
        **kwargs,
    )

    PlasmaState.build_posterior(
        diagnostics=[], priors=[prior], field_models=[linear_field]
    )

    # build some test parameters at which to evaluate the posterior
    param_dict = {"emission_linear_basis": sin(0.5 * R) + 1.0}
    param_array = PlasmaState.merge_parameters(param_dict)

    # evaluate the posterior gradient both analytically and numerically
    analytic_grad = posterior.gradient(param_array)
    numeric_grad = approx_fprime(xk=param_array, f=posterior.log_probability)

    # check that the fractional error between the gradients is small
    frac_err = numeric_grad / analytic_grad - 1
    assert abs(frac_err).max() < 1e-4
