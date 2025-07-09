Adding diagnostics to a MIDAS analysis
======================================

MIDAS abstracts the definition of a diagnostic likelihood into two parts:

* A ``LikelihoodFunction`` object, which holds the experimental measurements and
  uncertainties, and specifies a distribution used to model the uncertainties (e.g.
  Gaussian, logistic etc.)

* A ``DiagnosticModel`` object, which implements (or calls) the forward-model for
  the diagnostic, and specifies what information is required to evaluate the model
  predictions (e.g. the values of plasma fields like temperature or density at
  specific coordinates).

Specifying diagnostic models
----------------------------


Specifying likelihood functions
-------------------------------

