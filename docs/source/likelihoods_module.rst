.. _likelihoods-module:

The Likelihoods module
======================

.. _DiagnosticLikelihood-ref:

.. autoclass:: midas.likelihoods.DiagnosticLikelihood


Built-in likelihood functions
-----------------------------

.. _GaussianLikelihood-ref:

.. autoclass:: midas.likelihoods.GaussianLikelihood


.. autoclass:: midas.likelihoods.LogisticLikelihood


.. autoclass:: midas.likelihoods.CauchyLikelihood


Uncertainty models
------------------

.. autoclass:: ConstantUncertainty

.. autoclass:: LinearUncertainty


from midas.likelihoods.uncertainties import UncertaintyModel


Abstract base classes
---------------------

.. _LikelihoodFunction-ref:

.. autoclass:: midas.likelihoods.LikelihoodFunction
   :members: log_likelihood, predictions_derivative


.. autoclass:: midas.likelihoods.UncertaintyModel
   :members: get_uncertainties, get_uncertainties_and_jacobians
