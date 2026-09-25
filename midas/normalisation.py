from numpy import inf, isfinite, ndarray, zeros_like

from midas.posterior import Posterior

__all__ = ["NormalisedCost"]


class NormalisedCost:
    """
    Wrap the posterior cost and gradient functions so they operate on parameter values
    normalised to the interval [0, 1].

    :param posterior: \
        The posterior whose cost and gradient will be evaluated.

    :param bounds: \
        The lower and upper bounds of each model parameter as a 2D array with shape
        ``(n_parameters, 2)``.
    """

    def __init__(self, posterior: Posterior, bounds: ndarray):

        assert isinstance(posterior, Posterior)
        assert isinstance(bounds, ndarray)
        assert bounds.ndim == 2
        assert bounds.shape == (posterior.n_params, 2)
        assert (bounds[:, 1] > bounds[:, 0]).all()

        self.posterior = posterior
        self.scale = bounds[:, 1] - bounds[:, 0]
        self.shift = bounds[:, 0]
        self.normalised_bounds = zeros_like(bounds)
        self.normalised_bounds[:, 1] = 1.0

        assert (self.scale > 0).all()
        assert isfinite(self.scale).all()
        assert isfinite(self.shift).all()
        
        self.lowest_cost = inf
        self.best_theta = None

    def denormalise(self, normalised_point: ndarray) -> ndarray:
        """
        Convert normalised parameter values to their original scale.

        :param normalised_point: \
            The normalised parameter values as a 1D array.

        :return: \
            The parameter values on their original scale as a 1D array.
        """
        return normalised_point * self.scale + self.shift

    def normalise(self, theta: ndarray) -> ndarray:
        """
        Convert parameter values to the interval [0, 1].

        :param theta: \
            The parameter values on their original scale as a 1D array.

        :return: \
            The normalised parameter values as a 1D array.
        """
        return (theta - self.shift) / self.scale

    def cost(self, normalised_point: ndarray) -> float:
        """
        Calculate the posterior cost for a set of normalised parameter values.

        The lowest cost encountered and its corresponding parameter values on the
        original scale are stored in ``lowest_cost`` and ``best_theta`` respectively.

        :param normalised_point: \
            The normalised parameter values as a 1D array.

        :return: \
            The negative posterior log-probability.
        """
        theta = normalised_point * self.scale + self.shift
        c = self.posterior.cost(theta)
        if c < self.lowest_cost:
            self.lowest_cost = c
            self.best_theta = theta.copy()
        return c

    def cost_gradient(self, normalised_point: ndarray) -> ndarray:
        """
        Calculate the cost gradient with respect to normalised parameter values.

        :param normalised_point: \
            The normalised parameter values as a 1D array.

        :return: \
            The cost gradient with respect to the normalised parameter values as a
            1D array.
        """
        return (
            self.posterior.cost_gradient(
                normalised_point * self.scale + self.shift
            ) * self.scale
        )

    def component_cost(self, normalised_point: ndarray, component_name: str) -> float:
        """
        Calculate the cost for a specific component given a set of normalised parameter values.

        :param normalised_point: \
            The normalised parameter values as a 1D array.
            
        :param component_name: \
            The name of the component for which to calculate the cost.

        :return: \
            The cost for the specified component.
        """
        theta = normalised_point * self.scale + self.shift
        return -self.posterior.component_log_probability(theta, component_name)

    def component_cost_gradient(self, normalised_point: ndarray, component_name: str) -> ndarray:
        """Calculate a component's cost gradient in normalised coordinates."""
        return -self.posterior.component_gradient(
            self.denormalise(normalised_point), component_name
        ) * self.scale
