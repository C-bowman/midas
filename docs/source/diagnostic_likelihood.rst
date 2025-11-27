Adding diagnostics to a MIDAS analysis
======================================

To include a diagnostic in a MIDAS analysis, we need to create a :ref:`DiagnosticLikelihood <DiagnosticLikelihood-ref>`
object. MIDAS abstracts the definition of a diagnostic likelihood into two parts:


* A :ref:`DiagnosticModel <DiagnosticModel-ref>` object, which implements (or calls)
  the forward-model for the diagnostic, and specifies what information is required to
  evaluate the model predictions (e.g. the values of plasma fields like temperature or
  density at specific coordinates).


* A :ref:`LikelihoodFunction <LikelihoodFunction-ref>` object, which holds the experimental
  measurements and uncertainties, and specifies a distribution used to model the
  uncertainties (e.g. Gaussian, logistic etc.)


Specifying diagnostic models
----------------------------

For a diagnostic forward model to evaluate its predictions, it will require information
about the physical state of the system it is measuring, and potentially information
regarding the diagnostic itself, such as calibration values or background levels.

In MIDAS, the information required by models is grouped into two categories: 'parameters' and 'fields'.


Requesting field values
^^^^^^^^^^^^^^^^^^^^^^^
'fields' are the values of physical quantities at particular spatial coordinates.
For example, a model of a Thomson scattering diagnostic may require the values
of both the electron temperature and density at a set of :math:`(R, z)` coordinates.
To specify which field values are required by a diagnostic, we create instances of the
:ref:`FieldRequest <FieldRequest-ref>` class:

.. code-block:: python

    from numpy import linspace, full
    from midas import FieldRequest

    # example measurement positions for a Thomson-scattering diagnostic
    R_ts = linspace(0.3, 1.6, 131)
    z_ts = full(131, fill_value=0.01)

    # Request the electron temperature and density field values at these positions
    field_requests = [
        FieldRequest(name="T_e", coordinates={"radius": R_ts, "z": z_ts}),
        FieldRequest(name="n_e", coordinates={"radius": R_ts, "z": z_ts}),
    ]

Specifying required parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
'parameters' capture any other information required to evaluate the diagnostic model,
but are typically used to specify properties of the instrument itself. The parameters
required by a model are specified by creating instances of the
:ref:`ParameterVector <ParameterVector-ref>` class:

.. code-block:: python

    from midas import ParameterVector

    parameters = [
        ParameterVector(name="calibration_value", size=1),
        ParameterVector(name="background_line_coeffs", size=2),
    ]


Defining a diagnostic
^^^^^^^^^^^^^^^^^^^^^

Diagnostic models in ``midas`` are defined as classes, and must meet three
requirements:

* The class must inherit from the :ref:`DiagnosticModel <DiagnosticModel-ref>`
  abstract base-class, and therefore implement the required
  ``midas.models.DiagnosticModel.predictions`` and
  ``predictions_and_jacobians`` methods.

* Instances of the class must have a `parameters` instance attribute, which is
  a list containing :ref:`ParameterVector <ParameterVector-ref>` objects (or is empty).

* Instances of the class must have a `field_requests` instance attribute, which is
  a list containing :ref:`FieldRequest <FieldRequest-ref>` objects (or is empty).

For example, a simple straight-line model would not require any field values,
but would require parameters to define the gradient and offset:


.. code-block:: python

    from numpy import ndarray, ones_like
    from midas.models import DiagnosticModel


    class StraightLine(DiagnosticModel):
        def __init__(self, x_axis: ndarray):
            self.x = x_axis
            self.parameters = [
                ParameterVector(name="gradient", size=1),
                ParameterVector(name="y_intercept", size=1),
            ]
            self.field_requests = []

        def predictions(self, gradient: float, y_intercept: float) -> ndarray:
            return gradient * self.x + y_intercept

        def predictions_and_jacobians(self, gradient: float, y_intercept: float) -> tuple:
            predictions = gradient * self.x + y_intercept
            jacobians = {"gradient": self.x, "y_intercept": ones_like(self.x)}
            return predictions, jacobians



Specifying likelihood functions
-------------------------------
Likelihood functions are models for the uncertainties associated with the diagnostic
measurements. Classes for commonly used likelihood functions, such as the Gaussian and
logistic distributions, are available in the :ref:`midas.likelihoods <likelihoods-module>`
module.

Likelihood function classes encapsulate the experimental measurements and their
associated uncertainties, so these data must be passed as arguments. For example,
creating an instance of :ref:`GaussianLikelihood <GaussianLikelihood-ref>` could look like this:


.. code-block:: python

    from midas.likelihoods import GaussianLikelihood

    gaussian_likelihood = GaussianLikelihood(
        y_data=measurement_values,
        sigma=measurement_uncertainties,
    )


Creating a DiagnosticLikelihood
-------------------------------

Combining the previous examples of a straight-line model and a Gaussian likelihood,
we can create an instance of :ref:`DiagnosticLikelihood <DiagnosticLikelihood-ref>`:

.. code-block:: python

    from midas.likelihoods import DiagnosticLikelihood

    straight_line_model = StraightLine(x_axis=measurement_positions)

    straight_line_likelihood = DiagnosticLikelihood(
        diagnostic_model=straight_line_model,
        likelihood=gaussian_likelihood,
        name="straight_line"
    )

