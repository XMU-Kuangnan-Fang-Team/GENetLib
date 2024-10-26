from GENetLib.sim_data_scalar import sim_data_scalar
from GENetLib.sim_data_func import sim_data_func
from GENetLib.func_ge import func_ge
from GENetLib.grid_func_ge import grid_func_ge


def test_sim_data_scalar():
    scalar_survival_linear = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=1500, dim_E_Sparse=2, ytype='Survival', n_inter=30)
    assert scalar_survival_linear[0] is not None
    assert scalar_survival_linear[1] is not None


def test_sim_data_func_and_func_ge():
    func_continuous = sim_data_func(n=1500, m=30, ytype='Continuous', seed=123)
    assert func_continuous['X'] is not None 
    assert func_continuous['y'] is not None
    assert func_continuous['z'] is not None
    assert func_continuous['location'] is not None

    func_ge_res = func_ge(func_continuous['y'], func_continuous['z'], func_continuous['location'], func_continuous['X'], 'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100,10], Learning_Rate2=0.035, L2=0.01, Learning_Rate1=0.02, L=0.01, Num_Epochs=50, nbasis1=5, params1=4, Bsplines=5, norder1=4, model=None, split_type=1, ratio=[3, 1, 1], plot_res=True)
    assert func_ge_res is not None


def test_grid_func_ge():
    func_continuous = sim_data_func(n=1000, m=100, ytype='Continuous', seed=1)
    assert func_continuous['X'] is not None
    assert func_continuous['y'] is not None
    assert func_continuous['z'] is not None
    assert func_continuous['location'] is not None

    grid_func_ge_res = grid_func_ge(func_continuous['y'], func_continuous['z'], func_continuous['location'], func_continuous['X'], 'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100, 10], Learning_Rate2=[0.008, 0.009, 0.01], L2=[0.002, 0.003, 0.004, 0.005, 0.006], Learning_Rate1=[0.02, 0.03, 0.04, 0.05], L=[0.05, 0.06, 0.07, 0.08], Num_Epochs=100, nbasis1=7, params1=4, Bsplines=15, norder1=4, model=None, split_type=0, ratio=[7,3], plot_res=True)
    assert grid_func_ge_res is not None
