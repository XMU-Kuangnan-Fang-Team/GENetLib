from GENetLib.sim_data_scalar import sim_data_scalar
from GENetLib.grid_scalar_ge import grid_scalar_ge
import matplotlib
matplotlib.use('Agg')

def test_grid_scalar_ge():
  scalar_continuous = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Continuous', n_inter=30, seed = 111)
  scalar_binary = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Binary', n_inter=10, seed = 111)
  scalar_survival = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Survival', n_inter=10, linear = False, seed = 111)
  grid_scalar_ge_res_1 = grid_scalar_ge(data=scalar_continuous['data'], ytype='Continuous', dim_G=500, dim_E=5, haveGE=True, 
                                        num_hidden_layers=2, nodes_hidden_layer=[1000,100], 
                                        Learning_Rate2=[0.035,0.045], L2=[0.1], Learning_Rate1=[0.04, 0.05, 0.06], L=[0.08, 0.09], 
                                        Num_Epochs=50, t = 0.01, model = None, split_type = 0, ratio = [7, 3], 
                                        important_feature = True, plot = True, model_reg = None, isfunc = False)
  grid_scalar_ge_res_2 = grid_scalar_ge(data=scalar_binary['data'], ytype='Binary', dim_G=1500, dim_E=5, haveGE=False, 
                                        num_hidden_layers=1, nodes_hidden_layer=[50], 
                                        Learning_Rate2=[0.035,0.045], L2=[0.1], Learning_Rate1=[0.04, 0.05, 0.06], L=[0.08, 0.09], 
                                        Num_Epochs=50, t = 0.01, model = None, split_type = 1, ratio = [7, 2, 1], 
                                        important_feature = False, plot = False, model_reg = None, isfunc = False)
  grid_scalar_ge_res_3 = grid_scalar_ge(data=scalar_survival['data'], ytype='Survival', dim_G=500, dim_E=5, haveGE=False, 
                                        num_hidden_layers=2, nodes_hidden_layer=[1000,100], 
                                        Learning_Rate2=[0.035,0.045], L2=[0.1], Learning_Rate1=[0.04, 0.05, 0.06], L=[0.08, 0.09], 
                                        Num_Epochs=50, t = 0.01, model = None, split_type = 1, ratio = [7, 2, 1], 
                                        important_feature = False, plot = False, model_reg = None, isfunc = False)
  assert grid_scalar_ge_res_1 is not None
  assert grid_scalar_ge_res_2 is not None
  assert grid_scalar_ge_res_3 is not None
