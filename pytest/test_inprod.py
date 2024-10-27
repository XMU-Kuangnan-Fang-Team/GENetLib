from GENetLib.create_basis import create_bspline_basis, create_fourier_basis, create_expon_basis
from GENetLib.inprod import inprod


def test_inprod():
    fbasis1 = create_bspline_basis(rangeval=[0, 1], nbasis=5, breaks=[0, 0.5, 1])
    fbasis2 = create_expon_basis(rangeval=[0, 1], nbasis=2, ratevec=[0.5, 1.5])
    fbasis3 = create_fourier_basis(rangeval=[0, 1], nbasis=3)
    fbasis_1 = inprod(fbasis1)
    fbasis_2 = inprod(fbasis2)
    fbasis_3 = inprod(fbasis3)
    assert fbasis_1 is not None
    assert fbasis_2 is not None
    assert fbasis_3 is not None
