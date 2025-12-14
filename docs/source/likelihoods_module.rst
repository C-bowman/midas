.. _likelihoods-module:

The Likelihoods module
======================

.. _DiagnosticLikelihood-ref:

.. autoclass:: midas.likelihoods.DiagnosticLikelihood


Built-in likelihood functions
-----------------------------

Likelihood functions, which model the uncertainties associated with measured data
as various probability distributions, are implemented as classes in MIDAS which also
encapsulate the measured data and their estimated uncertainties.

.. _GaussianLikelihood-ref:

.. autoclass:: midas.likelihoods.GaussianLikelihood


.. autoclass:: midas.likelihoods.LogisticLikelihood


.. autoclass:: midas.likelihoods.CauchyLikelihood


Uncertainty models
------------------

As an alternative to passing fixed estimates of the uncertainties on measured data to
likelihood function classes, a model for the uncertainties can be given instead, allowing
them to vary.

.. autoclass:: midas.likelihoods.ConstantUncertainty

.. autoclass:: midas.likelihoods.LinearUncertainty



Abstract base classes
---------------------

.. _LikelihoodFunction-ref:

.. autoclass:: midas.likelihoods.LikelihoodFunction
   :members: log_likelihood, predictions_derivative


.. autoclass:: midas.likelihoods.UncertaintyModel
   :members: get_uncertainties, get_uncertainties_and_jacobians

