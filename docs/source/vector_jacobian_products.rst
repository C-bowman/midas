Vector-Jacobian products
========================

MIDAS propagates derivatives in reverse, from the scalar posterior log-probability
back to its parameters. Model interfaces therefore support vector-Jacobian products
(VJPs), which avoid constructing a complete Jacobian when a model can apply its
transpose directly.

How the interface works
-----------------------

For model outputs :math:`\boldsymbol{y} = f(\boldsymbol{x})`, let
:math:`J_x = \partial \boldsymbol{y} / \partial \boldsymbol{x}`. If the next stage of
the calculation supplies the vector
:math:`\boldsymbol{v} = \partial L / \partial \boldsymbol{y}`, the gradient needed for
the model input is

.. math::

    \frac{\partial L}{\partial \boldsymbol{x}}
    = \boldsymbol{v} J_x.

The methods in the VJP interface return the usual model result followed by a
``pullback`` function. Calling the pullback with :math:`\boldsymbol{v}` returns a
dictionary containing :math:`\boldsymbol{v} J_x` for every requested input.

MIDAS provides a default pullback for each model interface:

.. list-table::
   :header-rows: 1

   * - Model type
     - VJP method
     - Default Jacobian source
   * - :class:`~midas.models.DiagnosticModel`
     - ``predictions_and_pullback``
     - ``predictions_and_jacobians``
   * - :class:`~midas.models.FieldModel`
     - ``values_and_pullback``
     - ``values_and_jacobians``
   * - :class:`~midas.likelihoods.UncertaintyModel`
     - ``uncertainties_and_pullback``
     - ``uncertainties_and_jacobians``

Existing models therefore work without changes. Their default pullbacks calculate
``vector @ jacobian`` for each entry in the Jacobian dictionary. A model should
override its VJP method when it can calculate these products more efficiently, for
example when its Jacobian is diagonal, sparse, structured, or available through an
adjoint solver.

The returned pullback must:

* accept a 1D array with one value per model output;
* return every requested parameter or field under the same name used by the model;
* return each gradient as a 1D array, including gradients for scalar inputs; and
* calculate partial derivatives with the other explicit model inputs held fixed.

Pullbacks may be called more than once, so they should not consume or mutate captured
values. They can capture intermediate values from the forward calculation to avoid
repeating work.

A model using Jacobians
-----------------------

Consider a simple diagnostic with a scalar gain and one log-signal parameter per
measurement. Its predictions are

.. math::

    y_i = g \exp(s_i).

The Jacobian with respect to the log-signal is diagonal. The existing Jacobian
interface can represent the model as follows:

.. code-block:: python

    from numpy import diag, exp, ndarray

    from midas import Fields, Parameters, ParameterVector
    from midas.models import DiagnosticModel


    class CalibratedExponentialModel(DiagnosticModel):
        def __init__(self, size: int):
            self.parameters = Parameters(
                ParameterVector(name="gain", size=1),
                ParameterVector(name="log_signal", size=size),
            )
            self.fields = Fields()

        def predictions(
            self, gain: ndarray, log_signal: ndarray
        ) -> ndarray:
            return gain[0] * exp(log_signal)

        def predictions_and_jacobians(
            self, gain: ndarray, log_signal: ndarray
        ) -> tuple[ndarray, dict[str, ndarray]]:
            unit_response = exp(log_signal)
            predictions = gain[0] * unit_response
            jacobians = {
                "gain": unit_response,
                "log_signal": diag(predictions),
            }
            return predictions, jacobians

For :math:`N` measurements, ``diag(predictions)`` allocates an :math:`N \times N`
array even though only :math:`N` diagonal values contain information. The default
pullback then performs a dense vector-matrix product with that array.

Converting the model to use a VJP
---------------------------------

Keep ``predictions`` and ``predictions_and_jacobians`` for compatibility, and add a
direct pullback implementation:

.. code-block:: python

    from numpy import atleast_1d


    class VJPCalibratedExponentialModel(CalibratedExponentialModel):
        def predictions_and_pullback(
            self, gain: ndarray, log_signal: ndarray
        ):
            unit_response = exp(log_signal)
            predictions = gain[0] * unit_response

            def pullback(vector: ndarray) -> dict[str, ndarray]:
                return {
                    "gain": atleast_1d(vector @ unit_response),
                    "log_signal": vector * predictions,
                }

            return predictions, pullback

The ``gain`` result is :math:`\boldsymbol{v}` dotted with
:math:`\partial \boldsymbol{y} / \partial g`. For ``log_signal``, multiplying by the
diagonal Jacobian is just element-wise multiplication. This produces the same
gradients without creating the dense array, reducing this part of the calculation
from :math:`O(N^2)` storage and work to :math:`O(N)`.

During posterior-gradient evaluation, MIDAS calls ``predictions_and_pullback`` and
passes the derivative supplied by the likelihood to the returned function. The
Jacobian method is not called on this path. MIDAS then propagates any returned field
gradients through their field-model pullbacks and accumulates all contributions to
the posterior parameters.