import pytest
from numpy import array, eye, inf, linspace, nan, sin, allclose, concatenate
from scipy import sparse
from scipy.optimize import approx_fprime
from numpy.random import default_rng

from midas import ParameterVector
from midas.priors import GaussianProcessPrior, GaussianPrior, ExponentialPrior
from midas.priors import BetaPrior, SoftLimitPrior, SoftBoundsPrior
from midas.priors import LinearGaussianPrior
from midas.models.fields import PiecewiseLinearField, FieldRequest
from midas.state import PlasmaState
from midas import posterior

rng = default_rng(2391)


prior_test_setup = [
    (
        GaussianPrior,
        {
            "mean": rng.uniform(low=-1.0, high=1.0, size=16),
            "standard_deviation": rng.uniform(low=0.5, high=2.0, size=16),
        },
        {
            "numeric_arguments": ["mean", "standard_deviation"],
            "values_inside_support": rng.uniform(low=-5.0, high=5.0, size=16)
        }
    ),
    (
        ExponentialPrior,
        {
            "mean": rng.uniform(low=0.1, high=10.0, size=16),
        },
        {
            "numeric_arguments": ["mean"],
            "values_outside_support": linspace(-10.0, 0.0, 16),
            "values_inside_support": rng.uniform(low=0.1, high=20.0, size=16)
        },
    ),
    (
        BetaPrior,
        {
            "alpha": rng.uniform(low=0.3, high=3.0, size=16),
            "beta": rng.uniform(low=0.3, high=3.0, size=16),
            "limits": (-0.5, 2.5),
        },
        {
            "numeric_arguments": ["alpha", "beta"],
            "values_outside_support": concatenate([linspace(-10.0, -0.6, 8), linspace(2.6, 10.0, 8)]),
            "values_inside_support": rng.uniform(low=-0.2, high=2.2, size=16)
        },
    ),
    (
        SoftLimitPrior,
        {
            "upper_limit": rng.uniform(low=0., high=2.0, size=16),
            "standard_deviation": rng.uniform(low=0.5, high=2.0, size=16),
            "operator": None,
        },
        {
            "numeric_arguments": ["upper_limit", "standard_deviation"],
            "values_inside_support": rng.uniform(low=-5.0, high=5.0, size=16)
        }
    ),
    (
        SoftLimitPrior,
        {
            "upper_limit": rng.uniform(low=0., high=2.0, size=12),
            "standard_deviation": rng.uniform(low=0.5, high=2.0, size=12),
            "operator": rng.random(size=(12, 16)),
        },
        {
            "numeric_arguments": ["upper_limit", "standard_deviation"],
            "values_inside_support": rng.uniform(low=-5.0, high=5.0, size=16)
        }
    ),
    (
        LinearGaussianPrior,
        {
            "operator": rng.normal(scale=0.1, size=(12, 16)),
            "mean": rng.uniform(low=-1.0, high=1.0, size=12),
            "standard_deviation": rng.uniform(low=0.5, high=2.0, size=12),
        },
        {
            "numeric_arguments": ["mean", "standard_deviation"],
            "values_inside_support": rng.uniform(low=-5.0, high=5.0, size=16),
        },
    ),
    (
        SoftBoundsPrior,
        {
            "lower_limit": linspace(-2.0, -1.0, 16),
            "upper_limit": linspace(1.0, 2.0, 16),
            "standard_deviation": linspace(0.5, 2.0, 16),
            "operator": None,
        },
        {
            "numeric_arguments": ["lower_limit", "upper_limit", "standard_deviation"],
            "values_inside_support": linspace(-5.0, 5.0, 16),
        },
    ),
    (
        SoftBoundsPrior,
        {
            "lower_limit": linspace(-2.0, -1.0, 12),
            "upper_limit": linspace(1.0, 2.0, 12),
            "standard_deviation": linspace(0.5, 2.0, 12),
            "operator": eye(12, 16) - eye(12, 16, k=1),
        },
        {
            "numeric_arguments": ["lower_limit", "upper_limit", "standard_deviation"],
            "values_inside_support": array([-3.0, 3.0] * 8),
        },
    ),
]


@pytest.mark.parametrize(
    "prior_class, kwargs, info", prior_test_setup,
)
def test_bounded_support_priors_reject_invalid_values(
    prior_class, kwargs, info
):
    parameter_vector = ParameterVector(name="x", size=16)
    prior = prior_class(
        name="bounded_prior",
        parameter_vector=parameter_vector,
        **kwargs,
    )

    assert prior.probability(x=info["values_inside_support"]) > -1e50
    if "values_outside_support" in info:
        assert prior.probability(x=info["values_outside_support"]) == -1e50


@pytest.mark.parametrize("prior_class, kwargs, info", prior_test_setup)
def test_prior_validates_name(prior_class, kwargs, info):   
    parameter_vector = ParameterVector(name="x", size=16)

    with pytest.raises(TypeError, match="must be a string"):
        prior_class(
            name=None,
            parameter_vector=parameter_vector,
            **kwargs,
        )

    with pytest.raises(ValueError, match="must not be empty"):
        prior_class(
            name="",
            parameter_vector=parameter_vector,
            **kwargs,
        )


@pytest.mark.parametrize("prior_class, kwargs, info", prior_test_setup)
def test_prior_numeric_dtypes(prior_class, kwargs, info):
    parameter_vector = ParameterVector(name="x", size=16)

    for argument in info["numeric_arguments"]:
        testing_kwargs = kwargs.copy()
        testing_kwargs[argument] = kwargs[argument].astype(str)

        with pytest.raises(TypeError, match="real numeric values"):
            prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)

        testing_kwargs[argument] = kwargs[argument].astype(complex)

        with pytest.raises(TypeError, match="real numeric values"):
            prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)


@pytest.mark.parametrize("prior_class, kwargs, info", prior_test_setup)
def test_prior_numeric_finiteness(prior_class, kwargs, info):
    parameter_vector = ParameterVector(name="x", size=16)

    for argument in info["numeric_arguments"]:
        testing_kwargs = kwargs.copy()

        added_inf = kwargs[argument].copy()
        added_inf[-1] = inf
        testing_kwargs[argument] = added_inf

        with pytest.raises(ValueError, match="finite values"):
            prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)

        added_nan = kwargs[argument].copy()
        added_nan[-1] = nan
        testing_kwargs[argument] = added_nan

        with pytest.raises(ValueError, match="finite values"):
            prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)


@pytest.mark.parametrize("prior_class, kwargs, info", prior_test_setup)
def test_prior_numeric_shapes(prior_class, kwargs, info):
    parameter_vector = ParameterVector(name="x", size=16)

    for argument in info["numeric_arguments"]:
        testing_kwargs = kwargs.copy()
        testing_kwargs[argument] = kwargs[argument].reshape((1, -1))

        with pytest.raises(ValueError, match="array with dimension"):
            prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)

        testing_kwargs[argument] = kwargs[argument][:-2]

        with pytest.raises(ValueError, match="must have shape"):
            prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)


@pytest.mark.parametrize("prior_class, kwargs, info", prior_test_setup)
def test_prior_accepts_scalar_numeric_arguments(prior_class, kwargs, info):
    parameter_vector = ParameterVector(name="x", size=16)
    testing_kwargs = kwargs.copy()

    for argument in info["numeric_arguments"]:
        testing_kwargs[argument] = float(kwargs[argument][0])

    prior = prior_class(
        name="scalar_prior",
        parameter_vector=parameter_vector,
        **testing_kwargs,
    )

    assert prior.probability(x=info["values_inside_support"]) > -1e50


@pytest.mark.parametrize(
    "prior_class, kwargs, info",
    [setup for setup in prior_test_setup if setup[1].get("operator") is not None]
    )
def test_prior_operators(prior_class, kwargs, info):
    parameter_vector = ParameterVector(name="x", size=kwargs["operator"].shape[1])
    testing_kwargs = kwargs.copy()
    testing_kwargs["operator"] = kwargs["operator"][:, :-1]
    expected_error = AssertionError if prior_class is SoftLimitPrior else ValueError

    with pytest.raises(expected_error):
        prior_class(name="prior", parameter_vector=parameter_vector, **testing_kwargs)

    dense_prior = prior_class(name="prior", parameter_vector=parameter_vector, **kwargs)
    testing_kwargs["operator"] = sparse.csr_array(kwargs["operator"])
    sparse_prior = prior_class(
        name="prior", parameter_vector=parameter_vector, **testing_kwargs
    )
    values = info["values_inside_support"]

    assert sparse_prior.probability(x=values) == pytest.approx(
        dense_prior.probability(x=values)
    )
    assert allclose(
        sparse_prior.gradients(x=values)["x"], dense_prior.gradients(x=values)["x"]
    )


@pytest.mark.parametrize("target_type", ["parameter_vector", "field_request"])
def test_soft_bounds_prior_penalties(target_type):
    target = (
        ParameterVector(name="x", size=5)
        if target_type == "parameter_vector"
        else FieldRequest(name="x", coordinates={"radius": linspace(0.0, 1.0, 5)})
    )
    prior = SoftBoundsPrior(
        name="bounds",
        lower_limit=-1.0,
        upper_limit=1.0,
        standard_deviation=2.0,
        **{target_type: target},
    )
    values = array([-2.0, -1.0, 0.0, 1.0, 3.0])
    assert prior.probability(x=values) == pytest.approx(-0.625)
    assert allclose(prior.gradients(x=values)["x"], [0.25, 0.0, 0.0, 0.0, -0.5])
    assert allclose(
        prior.gradients(x=values)["x"],
        approx_fprime(values, lambda values: prior.probability(x=values)),
    )
    interior = linspace(-1.0, 1.0, 5)
    assert prior.probability(x=interior) == 0.0
    assert allclose(prior.gradients(x=interior)["x"], 0.0)


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
        field_request=request,
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


def test_gp_prior_rejects_coordinate_size_mismatch():
    parameter_vector = ParameterVector(name="profile", size=3)

    with pytest.raises(ValueError, match="one location for each"):
        GaussianProcessPrior(
            name="profile_gp",
            parameter_vector=parameter_vector,
            coordinates={"radius": linspace(0.0, 1.0, 2)},
        )


@pytest.mark.parametrize("prior_class, kwargs, info", prior_test_setup)
def test_unparameterized_priors(prior_class, kwargs, info):
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
        field_request=request,
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

    assert allclose(analytic_grad, numeric_grad)
