from GENetLib.sim_data_func import sim_data_func
from GENetLib.func_ge import func_ge
import matplotlib
matplotlib.use('Agg')

def test_func_ge():
  func_continuous = sim_data_func(n=500, m=30, ytype='Continuous', seed=123)
  func_binary = sim_data_func(n=500, m=30, ytype='Binary', seed=123)
  func_survival = sim_data_func(n=500, m=30, ytype='Survival', seed=123)
  func_ge_res_1 = func_ge(func_continuous['y'], func_continuous['z'], func_continuous['location'], func_continuous['X'], 
                          'Continuous', 'Bspline', num_hidden_layers=2, nodes_hidden_layer=[100,10], Learning_Rate2=0.035, 
                          L2=0.01, Learning_Rate1=0.02, L=0.01, Num_Epochs=50, nbasis1=5, params1=4, t=0.01, Bsplines=5, norder1=4, 
                          model=None, split_type=1, ratio=[3, 1, 1], plot_res=False)
  func_ge_res_2 = func_ge(func_binary['y'], func_binary['z'], func_binary['location'], func_binary['X'], 
                          'Binary', 'Fourier', num_hidden_layers=2, nodes_hidden_layer=[100,10], Learning_Rate2=0.035, 
                          L2=0.01, Learning_Rate1=0.02, L=0.01, Num_Epochs=50, nbasis1=5, params1=4, Bsplines=5, norder1=4, 
                          model=None, split_type=0, ratio=[7, 3], plot_res=True)
  func_ge_res_3 = func_ge(func_survival['y'], func_survival['z'], func_survival['location'], func_survival['X'], 
                          'Survival', 'Power', num_hidden_layers=2, nodes_hidden_layer=[100,10], Learning_Rate2=0.035, 
                          L2=0.01, Learning_Rate1=0.02, L=0.01, Num_Epochs=50, nbasis1=5, params1=[-1,-0.5,0,0.5,1], Bsplines=5, norder1=4, 
                          model=None, split_type=0, ratio=[7, 3], plot_res=True, plot_beta = False) 
  assert func_ge_res_1 is not None
  assert func_ge_res_2 is not None
  assert func_ge_res_3 is not None
