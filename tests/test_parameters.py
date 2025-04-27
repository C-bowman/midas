from midas.parameters import FieldRequest
from numpy import linspace


def test_field_request():
    # first test the __eq__ method by making various fields which should
    # evaluate as equal / not equal
    n_points = 128
    radius = linspace(0.5, 1.5, n_points)
    f1 = FieldRequest(name="ln_te", coordinates={"radius": radius})
    f2 = FieldRequest(name="ln_te", coordinates={"radius": radius})
    f3 = FieldRequest(name="ln_te", coordinates={"radius": radius + 1e-4})
    f4 = FieldRequest(name="ln_ne", coordinates={"radius": radius})

    assert f1 == f2
    assert f1 != f3
    assert f1 != f4
    assert f1.__hash__() == f2.__hash__()

    # now test set membership
    field_set = {f2, f3, f4}
    # f1 should be in this set as it is equal to f2:
    assert f1 in field_set

    # check the size attribute is correct
    assert f1.size == n_points