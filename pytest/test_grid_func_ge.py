from GENetLib.sim_data_func import sim_data_func
from GENetLib.grid_func_ge import grid_func_ge
import matplotlib
matplotlib.use('Agg')
def test_grid_func_ge():
  func_continuous = sim_data_func(n=500, m=30, ytype='Continuous', seed=123)
  func_ge_res = grid_func_ge(func_continuous['y'], func_continuous['X'], func_continuous['location'], func_continuous['Z'], 
                             'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100,10], num_epochs=50,
                             learning_rate1=[0.02], learning_rate2=[0.009, 0.01], nbasis1=5, params1=4, lambda1=0.01,
                             lambda2=[0.01], Lambda=[0.002, 0.003], Bsplines=5, norder1=4, 
                             model=None, split_type=1, ratio=[3, 1, 1], plot_res=False)
  assert test_grid_func_ge is not None
