from GENetLib.sim_data_func import sim_data_func
from GENetLib.grid_func_ge import grid_func_ge

def test_grid_func_ge():
    func_continuous = sim_data_func(n=10, m=30, ytype='Continuous', seed=123)
    grid_func_ge_res = grid_func_ge(func_continuous['y'], func_continuous['z'], func_continuous['location'], 
                                    func_continuous['X'], 'Continuous', 'Bspline', num_hidden_layers=1, nodes_hidden_layer=[2], 
                                    Learning_Rate2=[0.009], L2=[0.004], 
                                    Learning_Rate1=[0.04], L=[0.05, 0.06], 
                                    Num_Epochs=1, nbasis1=5, params1=4, Bsplines=5, norder1=4, model=None, 
                                    split_type=0, ratio=[7,3], plot_res=False)
    assert grid_func_ge_res is not None
