from GENetLib.sim_data import sim_data_func
from GENetLib.func_ge import func_ge
import matplotlib
matplotlib.use('Agg')

def test_func_ge():
  func_continuous = sim_data_func(n=500, m=30, ytype='Continuous', seed=123)
  func_binary = sim_data_func(n=500, m=30, ytype='Binary', seed=123)
  func_survival = sim_data_func(n=500, m=30, ytype='Survival', seed=123)
  func_ge_res_1 = func_ge(func_continuous['y'], func_continuous['X'], func_continuous['location'], func_continuous['Z'],
                          'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100,10], num_epochs=50,
                          learning_rate1=0.02, learning_rate2=0.035, nbasis1=5, params1=4, lambda2=0.01, Lambda=0.01,
                          threshold=0.01, Bsplines=5, norder1=4, model=None, split_type=1, ratio=[3, 1, 1], plot_res=False)
  func_ge_res_2 = func_ge(func_binary['y'], func_binary['X'], func_binary['location'], func_binary['Z'], 
                          'Binary', 'Fourier', num_hidden_layers=2, nodes_hidden_layer=[100,10], num_epochs=50,
                          learning_rate1=0.02, learning_rate2=0.035, nbasis1=5, params1=4, lambda2=0.01, Lambda=0.01,
                          threshold=0.01, Bsplines=5, norder1=4, model=None, split_type=0, ratio=[7,3], plot_res=True)
  func_ge_res_3 = func_ge(func_survival['y'], func_survival['X'], func_survival['location'], func_survival['Z'], 
                          'Survival', 'Power', num_hidden_layers=2, nodes_hidden_layer=[100,10], num_epochs=50,
                          learning_rate1=0.02, learning_rate2=0.035, nbasis1=5, params1=[-1,-0.5,0,0.5,1], lambda2=0.01, Lambda=0.01,
                          threshold=0.01, Bsplines=5, norder1=4, model=None, split_type=1, ratio=[3, 1, 1], plot_res=False, plot_beta = False)
  assert func_ge_res_1 is not None
  assert func_ge_res_2 is not None
  assert func_ge_res_3 is not None
