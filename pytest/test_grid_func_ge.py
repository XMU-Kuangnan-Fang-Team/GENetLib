import pytest
from GENetLib.sim_data_func import sim_data_func
from GENetLib.grid_func_ge import grid_func_ge

@pytest.fixture
def func_continuous_data():
    return sim_data_func(n=1500, m=30, ytype='Continuous', seed=123)

def test_grid_func_ge(func_continuous_data):
    func_continuous = func_continuous_data
    grid_func_ge_res = grid_func_ge(func_continuous['y'], func_continuous['z'], func_continuous['location'], func_continuous['X'], 'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100, 10], Learning_Rate2=[0.008, 0.009, 0.01], L2=[0.002, 0.003, 0.004, 0.005, 0.006], Learning_Rate1=[0.02, 0.03, 0.04, 0.05], L=[0.05, 0.06, 0.07, 0.08], Num_Epochs=100, nbasis1=7, params1=4, Bsplines=15, norder1=4, model=None, split_type=0, ratio=[7,3], plot_res=True)
    assert grid_func_ge_res is not None
