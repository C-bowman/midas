.. _posterior-ref:

Constructing and evaluating a posterior
=======================================

The :mod:`midas.posterior` module defines posterior components, construction, vector
parameterisation, and evaluation. The ``Posterior`` class owns one independent
posterior distribution and its methods can be passed directly to optimisers and
samplers.

.. autofunction:: midas.build_posterior

.. autoclass:: midas.Posterior
   :members: log_probability, gradient, split_parameters, merge_parameters, split_samples, build_bounds, cost, cost_gradient, component_log_probabilities, get_model_predictions, sample_model_predictions, sample_field_values
