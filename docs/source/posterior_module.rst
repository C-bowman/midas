.. _posterior-ref:

Evaluating a posterior
======================

Posterior evaluations are methods of the instance returned by
:func:`midas.build_posterior`.

.. autoclass:: midas.Posterior
	:no-index:
	:members: log_probability, gradient, cost, cost_gradient, component_log_probabilities, get_model_predictions, sample_model_predictions, sample_field_values

Normalising the cost function for optimisation
----------------------------------------------

.. autoclass:: midas.posterior.NormalisedCost
	:members: denormalise, normalise, cost, cost_gradient