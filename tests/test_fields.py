import pytest
from numpy import linspace, allclose, log
from numpy.linalg import solve
from numpy.random import default_rng
from midas.models.fields import (
    PiecewiseLinearField,
    CubicSplineField,
    BSplineField,
    ExSplineField,
)
from midas.parameters import FieldRequest


@pytest.mark.parametrize(
    "field_model_class",
    [PiecewiseLinearField, CubicSplineField, BSplineField, ExSplineField],
)
def test_1d_field_interpolation(field_model_class):
    axis = linspace(0, 1, 32)
    field_model = field_model_class(
        field_name="emission",
        axis_name="radius",
        axis=axis,
    )

    # Evaluate interpolation at reproducible positions that are not basis knots.
    rng = default_rng(2391)
    random_positions = rng.uniform(low=axis[0], high=axis[-1], size=100)
    axis_request = FieldRequest(name="emission", coordinates={"radius": axis})
    request = FieldRequest(name="emission", coordinates={"radius": random_positions})

    test_line = lambda x: 0.2 * x + 0.3
    # Fit each model's parameterisation to the same line at the basis knots.
    parameter_values = solve(
        field_model.get_basis(axis_request), test_line(axis)
    )

    interpolated_values = field_model.get_values(
        parameters={field_model.param_name: parameter_values}, field=request
    )
    # ExSplineField exponentiates its spline, so compare in its latent log space.
    if isinstance(field_model, ExSplineField):
        interpolated_values = log(interpolated_values)

    assert allclose(interpolated_values, test_line(random_positions), atol=5e-4)
