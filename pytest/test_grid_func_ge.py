from GENetLib.sim_data import sim_data_func
from GENetLib.grid_func_ge import grid_func_ge
from GENetLib.predict_ge import predict_func
import matplotlib
matplotlib.use('Agg')
def test_grid_func_ge():
  func_continuous = sim_data_func(n=500, m=30, ytype='Continuous', seed=123)
  func_ge_res = grid_func_ge(func_continuous['y'], func_continuous['X'], func_continuous['location'], func_continuous['Z'], 
                             'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100,10], num_epochs=50,
                             learning_rate1=[0.02], learning_rate2=[0.009, 0.01], nbasis1=5, params1=4, lambda1=0.01,
                             lambda2=[0.01], Lambda=[0.002, 0.003], threshold=0.01, Bsplines=5, norder1=4, 
                             model=None, split_type=1, ratio=[3, 1, 1], plot_res=False)
  pred1 = predict_func(func_ge_res, func_continuous['y'], 'Continuous', func_continuous['X'], func_continuous['Z'], func_continuous['location'], Bsplines = 5)
  assert test_grid_func_ge is not None
  assert pred1 is not None
  ytype = 'Continuous'
  num_hidden_layers = 2
  nodes_hidden_layer = [100, 10]
  learning_rate2 = [0.008, 0.009, 0.01]
  Lambda = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
  learning_rate1 = [0.02, 0.03, 0.04, 0.05]
  lambda2 = [0.05, 0.06, 0.07, 0.08]
  num_epochs = 100
  nbasis1 = 7
  params1 = 4
  func_continuous = sim_data_func(n = 1000, m = 30, ytype = 'Continuous', input_type = 'func', seed = 123)
  y = func_continuous['y']
  Z = func_continuous['Z']
  location = func_continuous['location']
  X = func_continuous['X']
  grid_func_ge_res = grid_func_ge(y, X, location, Z, ytype, 'Bspline', 
                                  num_hidden_layers, nodes_hidden_layer, num_epochs, learning_rate1, learning_rate2, 
                                  nbasis1, params1, lambda1 = None, lambda2 = lambda2, Lambda = Lambda,
                                  Bsplines = 5, norder1 = 4, model = None, split_type = 1, ratio = [3, 1, 1],
                                  plot_res = False, plot_beta = True)
  pred2 = predict_func(grid_func_ge_res, y, ytype, X, Z, location, Bsplines = 5)
  assert grid_func_ge_res is not None
  assert pred2 is not None
