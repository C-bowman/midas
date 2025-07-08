# MIDAS
MIDAS is a framework for Bayesian and integrated data analysis. Documentation and
examples are under construction! Some key features of MIDAS are:

### Use diagnostic models from any source
MIDAS is designed to work with any diagnostic model which can by called from within Python,
and does not require models to be implemented within a specific framework. Instead,
MIDAS provides tools to create a lightweight wrapper around external forward-models
which allows them to interface with MIDAS.

### Efficient inference through analytic propagation of derivatives
Efficient MAP estimation and MCMC sampling in inference problems with ~20 or more free
parameters relies heavily on the ability to calculate the derivative of the posterior
log-probability with respect to those parameters.

Given the Jacobian of a diagnostic model (i.e. the derivatives of the model predictions
with respect to the model inputs) MIDAS will automatically propagate those derivatives
through the subsequent steps in calculating the posterior log-probability, so the
gradient of the posterior log-probability can be calculated analytically.

This allows MIDAS tackle large-scale problems with hundreds or thousands of free
parameters, or to solve smaller problems quickly and routinely.

### Easy interfacing to the Python scientific software ecosystem
MIDAS is designed to be used easily with external libraries, for example
using optimisers from [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
to maximise the posterior log-probability, or MCMC samplers from 
[`inference-tools`](https://github.com/C-bowman/inference-tools) to sample from the posterior.

### Modularity to allow easy exchange of models
Analysis in MIDAS is built from three types of models:
 - Diagnostic forward-models which make predictions of diagnostic signals.
 - Likelihood functions which model the uncertainties on measured data.
 - Plasma field models which give a parametrised description of the plasma state.

Each of these model types have interfaces defined by an associated abstract base-class,
which allows them to communicate with the framework. This abstraction means that
models can be easily swapped in and out of the analysis without requiring code changes.

For example, a forward-model for a Thomson-scattering diagnostic is able to request
the values of the electron temperature and density from their associated field models,
but is completely independent of the specific choice of parametrisation for those fields.

## Installation

MIDAS is available from [PyPI](https://pypi.org/project/midas-fusion/), 
so can be easily installed using [pip](https://pip.pypa.io/en/stable/) as follows:
```bash
pip install midas-fusion
```

## Documentation
 Documentation is still under construction - currently available docs can be found at
 [midas-fusion.readthedocs.io](https://midas-fusion.readthedocs.io/en/latest/).