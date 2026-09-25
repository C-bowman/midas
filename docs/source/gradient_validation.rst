.. _gradient-validation-ref:

Validating posterior gradients
==============================

Analytic posterior gradients can be checked against centred finite differences
after constructing the posterior with ``PlasmaState.build_posterior``. Pass the
same parameter bounds used for optimisation to ``validate_gradient``:

.. code-block:: python

   from midas.validation import validate_gradient

   report = validate_gradient(parameter_bounds=bounds, n_samples=5)
   report.print_report()

   if not report:
       raise RuntimeError("Posterior gradient validation failed")

The bounds are used to normalise every parameter before sampling. Validation
points are drawn from the central 80% of the normalised range, avoiding checks
directly on parameter boundaries. Each posterior component is checked separately
for every parameter vector.

The validator compares gradient magnitude and direction independently. It starts
with a centred finite-difference step close to the cube root of floating-point
machine precision, then tests progressively smaller steps while both errors
improve. ``initial_step`` can be overridden for noisy or poorly scaled models.

``GradientReport`` evaluates as true only when every sampled check passes. Its
``results`` attribute provides the maximum errors and pass count for each
component/parameter pair. ``print_report`` uses coloured output by default;
pass ``color=False`` when writing to a file or unsupported terminal.


Validation API
--------------

.. autofunction:: midas.validation.validate_gradient


Gradient report
~~~~~~~~~~~~~~~

``validate_gradient`` returns a :class:`GradientReport` instance:

.. currentmodule:: midas.validation

.. py:class:: GradientReport

   Summarises the results of validating posterior component gradients. A report
   evaluates as true when every sampled component/parameter gradient passes both
   error tolerances.

   .. py:attribute:: success
      :type: bool

      Whether every gradient check passed.

   .. py:attribute:: results
      :type: dict

      Nested mapping from component names to parameter names and their maximum
      direction error, maximum magnitude error, and number of passing samples.

   .. py:attribute:: n_samples
      :type: int

      Number of validation points sampled for each gradient.

   .. automethod:: GradientReport.print_report