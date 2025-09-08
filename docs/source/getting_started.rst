Getting started
===============

Installation
------------

MIDAS is available from `PyPI <https://pypi.org/project/midas-fusion/>`_,
so can be easily installed using `pip <https://pip.pypa.io/en/stable/>`_ as follows:

.. code-block:: bash

   pip install midas-fusion

If pip is not available, you can clone from the GitHub `source repository <https://github.com/C-bowman/midas>`_.


Structure of a MIDAS analysis
-----------------------------

The high-level structure of a MIDAS analysis can be broken down as:

* Create a ``DiagnosticLikelihood`` object for each diagnostic which is to be included in the
  analysis.
* Specify the prior distribution (or its components) as ``BasePrior`` objects.
* Specify models for the plasma fields in the analysis by calling the
  ``PlasmaState.specify_field_models()`` function.
* Build the parametrisation for the posterior distribution by calling the
  ``PlasmaState.build_parametrisation()`` function.
* Use the functions in the ``midas.posterior`` module to evaluate the posterior
  distribution, allowing for MAP estimation or sampling.

In subsequent pages we will cover each of these steps in more detail.

Jupyter notebook examples
-------------------------

Annotated example code is available as a jupyter notebook in our
`soft X-ray emission toy example <https://github.com/C-bowman/midas-examples/blob/main/sxr_example/sxr_example_notebook.ipynb>`_.
Additional example notebooks will be added as development progresses!
