import pytest
from numpy import linspace, exp
from midas.transforms import PsiTransform


def test_psi_transform():
    # create some testing psi data
    n = 64
    R = linspace(0.0, 2.0, n)
    z = linspace(-2.0, 2.0, n)
    rho_sqr = ((R[:, None] - 1.0) / 0.2)**2 + (z[None, :] / 0.4)**2
    psi = 1 - exp(-0.5*rho_sqr)

    # build the transform
    psi_transform = PsiTransform(
        R=R, z=z, psi=psi
    )

    # evaluate a slice of psi
    R_slice = linspace(0, 2.0, 128)
    z_slice = linspace(-1.5, 1.0, 128)
    psi_slice = psi_transform(dict(R=R_slice, z=z_slice))["psi"]

    with pytest.raises(AssertionError):
        psi_transform = PsiTransform(
            R=R.reshape([R.size, 1]), z=z, psi=psi
        )

    with pytest.raises(AssertionError):
        R2 = linspace(0.0, 2.0, n + 1)
        psi_transform = PsiTransform(R=R2, z=z, psi=psi)

    with pytest.raises(AssertionError):
        psi_transform = PsiTransform(R=[R], z=z, psi=psi)
