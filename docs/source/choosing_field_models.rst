Specifying models for the plasma fields
=======================================

For each named field specified by a :ref:`FieldRequest <FieldRequest-ref>` object in the
analysis, a corresponding model for the field is required. Field models are classes
which inherit from the :ref:`FieldModel <FieldModel-ref>` abstract base-class.

Like :ref:`DiagnosticModel <DiagnosticModel-ref>` objects, :ref:`FieldModel <FieldModel-ref>`
object also specify their required parameters through a ``parameters`` instance attribute
containing :ref:`ParameterVector <ParameterVector-ref>` objects.

The model for a given field cannot depend on the values of another field however,
so field models do not make use of :ref:`FieldRequest <FieldRequest-ref>` objects.

For example, if we could model the 1D profile of a field as a Gaussian function
using the following field model:


.. code-block:: python

    from numpy import exp
    from midas.fields import FieldModel


    class GaussianField(FieldModel):
        def __init__(self, field_name: str, axis: ndarray, axis_name: str):
            self.name = field_name
            self.axis = axis
            self.axis_name = axis_name

            # set up names for the parameters based on the given field name
            self.amplitude = f"{field_name}_amplitude"
            self.centre = f"{field_name}_centre"
            self.width = f"{field_name}_width"
            # create the parameter set
            self.parameters = [
                ParameterVector(name=self.amplitude, size=1),
                ParameterVector(name=self.centre, size=1),
                ParameterVector(name=self.width, size=1),
            ]

        def get_values(
            self, parameters: dict[str, ndarray], field: FieldRequest
        ) -> ndarray:
            # retrieve the coordinate values from the given FieldRequest
            coords = field.coordinates[self.axis_name]
            # build the Gaussian profile
            z = (coords - parameters[self.centre]) / parameters[self.width]
            field_values = parameters[self.amplitude] * exp(-0.5*z**2)
            return field_values

        def get_values_and_jacobian(
            self, parameters: dict[str, ndarray], field: FieldRequest
        ) -> tuple[ndarray, dict[str, ndarray]]:
            # retrieve the coordinate values from the given FieldRequest
            coords = field.coordinates[self.axis_name]
            # build the gaussian profile
            z = (coords - parameters[self.centre]) / parameters[self.width]
            g = exp(-0.5*z**2)
            field_values = parameters[self.amplitude] * g
            # calculate some derivatives required for the jacobians
            dg_dz = -z * g
            dz_dw = -z / parameters[self.width]
            dz_dc = coords / parameters[self.width]
            # build the jacobians dictionary
            jacobians = {
                self.amplitude: g,
                self.centre: dg_dz * dz_dc,
                self.width: dg_dz * dz_dw,
            }
            return field_values, jacobians