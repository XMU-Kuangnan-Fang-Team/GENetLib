import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from GENetLib.plot_fd import plot_fd
from GENetLib.fd_chk import fd_chk
from GENetLib.basis_fd import basis_fd


def test_plot_fd():
    plot_fd(fd_chk(basis_fd(btype="fourier", rangeval=[0, 1], nbasis=4, params=[2 * np.pi]))[1])
