from concurrent.futures import ThreadPoolExecutor

import pytest
from numpy import array, linspace
from numpy.testing import assert_allclose

from midas import Fields, Parameters, build_posterior
from midas.likelihoods import Diagnostic, GaussianLikelihood
from midas.priors import BasePrior

from utilities import Polynomial, StraightLine


def build_line_diagnostic(name="line"):
    x, y, sigma = StraightLine.testing_data()
    return Diagnostic(
        diagnostic_model=StraightLine(x),
        likelihood=GaussianLikelihood(y, sigma),
        name=name,
    )


def test_independent_posteriors_survive_later_and_failed_builds():
    line = build_posterior([build_line_diagnostic()], [], [])
    line_theta = line.merge_parameters({"gradient": 2.0, "y_intercept": -0.5})
    expected = line.log_probability(line_theta)

    x, y, sigma = StraightLine.testing_data()
    polynomial = build_posterior([
        Diagnostic(
            diagnostic_model=Polynomial(x, order=2),
            likelihood=GaussianLikelihood(y, sigma),
            name="polynomial",
        )
    ], [], [])

    assert line.n_params == 2
    assert polynomial.n_params == 3
    assert line.log_probability(line_theta) == expected

    with pytest.raises(ValueError, match="unique name"):
        build_posterior(
            [build_line_diagnostic("duplicate"), build_line_diagnostic("duplicate")],
            [],
            [],
        )

    assert line.log_probability(line_theta) == expected


class NestedEvaluationPrior(BasePrior):
    def __init__(self, nested_posterior, nested_theta):
        self.name = "nested"
        self.parameters = Parameters(("outer", 1))
        self.fields = Fields()
        self.nested_posterior = nested_posterior
        self.nested_theta = nested_theta

    def probability(self, **values):
        self.nested_posterior.log_probability(self.nested_theta)
        return -0.5 * (values["outer"] ** 2).sum()

    def gradients(self, **values):
        return {"outer": -values["outer"]}


def test_nested_evaluation_does_not_replace_outer_values():
    inner = build_posterior([build_line_diagnostic()], [], [])
    inner_theta = inner.merge_parameters({"gradient": 1.5, "y_intercept": 0.2})
    outer = build_posterior(
        [], [NestedEvaluationPrior(inner, inner_theta)], []
    )

    theta = array([0.4])
    assert outer.log_probability(theta) == pytest.approx(-0.08)
    assert_allclose(outer.gradient(theta), array([-0.4]))


def test_component_can_be_reused_by_independent_posteriors():
    diagnostic = build_line_diagnostic()
    first = build_posterior([diagnostic], [], [])
    second = build_posterior([diagnostic], [], [])
    theta = array([2.0, -0.5])

    assert first.components[0] is second.components[0]
    assert first.log_probability(theta) == second.log_probability(theta)
    assert_allclose(first.gradient(theta), second.gradient(theta))


def test_one_posterior_supports_concurrent_evaluations():
    posterior = build_posterior([build_line_diagnostic()], [], [])
    points = [array([gradient, -0.5]) for gradient in linspace(0.5, 3.0, 20)]
    expected = [posterior.log_probability(point) for point in points]

    with ThreadPoolExecutor(max_workers=4) as executor:
        actual = list(executor.map(posterior.log_probability, points))

    assert_allclose(actual, expected)