import pytest
from numpy import array, nan
from scipy.optimize import approx_fprime
from midas.likelihoods import GaussianLikelihood, LogisticLikelihood, CauchyLikelihood


@pytest.mark.parametrize(
    "likelihood",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
def test_likelihood_validation(likelihood):
    y = array([1., 3., 4.])
    sig = array([5., 5., 3.])

    # check the type validation
    with pytest.raises(TypeError):
        likelihood(y, [s for s in sig])

    # check array shape validation
    with pytest.raises(ValueError):
        likelihood(y[:-1], sig)

    with pytest.raises(ValueError):
        likelihood(y, sig.reshape([3, 1]))

    # check finite values validation
    y[1] = nan
    with pytest.raises(ValueError):
        likelihood(y, sig)


@pytest.mark.parametrize(
    "likelihood",
    [GaussianLikelihood, LogisticLikelihood, CauchyLikelihood],
)
def test_likelihoods_predictions_gradient(likelihood):
    test_values = array([3.58, 2.11, 7.89])
    y = array([1., 3., 4.])
    sig = array([5., 5., 3.])
    func = likelihood(y, sig)

    analytic_grad, _ = func.derivatives(predictions=test_values)
    numeric_grad = approx_fprime(f=func.log_likelihood, xk=test_values)
    max_abs_err = abs(analytic_grad - numeric_grad).max()
    assert max_abs_err < 1e-6