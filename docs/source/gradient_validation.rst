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

For example, a successful report can be seen in the
`Z-effective example notebook
<https://github.com/C-bowman/midas-examples/blob/main/z-eff/z_effective_inference.ipynb>`_
and is formatted as:

.. role:: gradient-report-pass
   :class: gradient-report-pass

.. role:: gradient-report-component
   :class: gradient-report-component

.. parsed-literal::

   Gradient validation: :gradient-report-pass:`PASS`

   :gradient-report-component:`[brem_diagnostic]`
       Parameter           | Status | Max direction error | Max magnitude error | Pass rate
       --------------------+--------+---------------------+---------------------+----------
       ln_ne_bspline_basis | :gradient-report-pass:`PASS`   |            8.88e-10 |            8.49e-10 |    100.0%
       ln_te_bspline_basis | :gradient-report-pass:`PASS`   |            1.12e-08 |            8.73e-10 |    100.0%
       z_eff_cubic_spline  | :gradient-report-pass:`PASS`   |            1.13e-08 |            7.07e-10 |    100.0%

   :gradient-report-component:`[te_diagnostic]`
       Parameter           | Status | Max direction error | Max magnitude error | Pass rate
       --------------------+--------+---------------------+---------------------+----------
       ln_ne_bspline_basis | :gradient-report-pass:`PASS`   |                   0 |                   0 |    100.0%
       ln_te_bspline_basis | :gradient-report-pass:`PASS`   |            2.24e-10 |            8.38e-11 |    100.0%
       z_eff_cubic_spline  | :gradient-report-pass:`PASS`   |                   0 |                   0 |    100.0%

   :gradient-report-component:`[ne_diagnostic]`
       Parameter           | Status | Max direction error | Max magnitude error | Pass rate
       --------------------+--------+---------------------+---------------------+----------
       ln_ne_bspline_basis | :gradient-report-pass:`PASS`   |            4.13e-10 |            1.85e-10 |    100.0%
       ln_te_bspline_basis | :gradient-report-pass:`PASS`   |                   0 |                   0 |    100.0%
       z_eff_cubic_spline  | :gradient-report-pass:`PASS`   |                   0 |                   0 |    100.0%

   :gradient-report-component:`[te_monotonicity_prior]`
       Parameter           | Status | Max direction error | Max magnitude error | Pass rate
       --------------------+--------+---------------------+---------------------+----------
       ln_ne_bspline_basis | :gradient-report-pass:`PASS`   |                   0 |                   0 |    100.0%
       ln_te_bspline_basis | :gradient-report-pass:`PASS`   |            1.33e-10 |            1.15e-10 |    100.0%
       z_eff_cubic_spline  | :gradient-report-pass:`PASS`   |                   0 |                   0 |    100.0%