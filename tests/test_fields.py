from numpy import linspace, allclose
from numpy.random import default_rng
from midas.fields import PiecewiseLinearField
from midas.parameters import FieldRequest


def test_piecewise_linear_field():
    # build a linear field
    R = linspace(1, 10, 10)
    linear_field = PiecewiseLinearField(
        field_name="emission",
        axis_name="radius",
        axis=R
    )

    # generate some random positions at which to request field values
    rng = default_rng(2391)
    random_positions = rng.uniform(low=1, high=10, size=30)
    request = FieldRequest(name="emission", coordinates={"radius": random_positions})

    # if we use a straight line as the test function, the interpolation should be exact
    test_line = lambda x: 5.12 * x + 0.74
    basis_values = test_line(R)
    interpolation_targets = test_line(random_positions)

    # request the values and check they match the targets
    interpolated_values = linear_field.get_values(
        parameters=basis_values,
        field=request
    )

    assert allclose(interpolated_values, interpolation_targets)
