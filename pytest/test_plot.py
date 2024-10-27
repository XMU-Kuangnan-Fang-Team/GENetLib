import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from GENetLib.plot_fd import plot_fd
from GENetLib.fd_chk import fd_chk
from GENetLib.basis_fd import basis_fd
from GENetLib.plot_rawdata import plot_rawdata
from GENetLib.sim_data_func import sim_data_func


def test_plot_fd():
    plot_fd(fd_chk(basis_fd(btype="fourier", rangeval=[0, 1], nbasis=4, params=[2 * np.pi]))[1])
    assert plt.gcf() is not None

def test_plot_rawdata():
    func_continuous = sim_data_func(n=1500, m=30, ytype='Continuous', seed=123)
    plot_rawdata(func_continuous['location'], func_continuous['X'], color = None, pch = 4, cex = 0.9):
    assert plt.gcf() is not None

    
