from GENetLib.sim_data import sim_data_scalar
from GENetLib.grid_scalar_ge import grid_scalar_ge
from GENetLib.predict_ge import predict_scalar
import matplotlib
matplotlib.use('Agg')

def test_grid_scalar_ge():
  scalar_continuous = sim_data_scalar(rho_G=0.75, rho_E=0.3, dim_G=500, dim_E=5, n=1000, dim_E_Sparse=2, ytype='Continuous', n_inter=30, seed = 111)
  scalar_binary = sim_data_scalar(rho_G=0.75, rho_E=0.3, dim_G=500, dim_E=5, n=1000, dim_E_Sparse=2, ytype='Binary', n_inter=10, seed = 111)
  scalar_survival = sim_data_scalar(rho_G=0.75, rho_E=0.3, dim_G=500, dim_E=5, n=1000, dim_E_Sparse=2, ytype='Survival', n_inter=10, linear = False, seed = 111)
  grid_scalar_ge_res_1 = grid_scalar_ge(scalar_continuous['y'], scalar_continuous['G'], scalar_continuous['E'], ytype='Continuous', 
                                        num_hidden_layers=2, nodes_hidden_layer=[1000,100], num_epochs=50,
                                        learning_rate1=[0.04, 0.05, 0.06], learning_rate2=[0.035,0.045],
                                        lambda1 = None, lambda2 = [0.08, 0.09], Lambda = [0.1],
                                        threshold = 0.01, model = None, split_type = 0, ratio = [7, 3], 
                                        important_feature = True, plot = True, model_reg = None, isfunc = False)
  y = scalar_survival['y']
  G = scalar_survival['G']
  E = scalar_survival['E']
  ytype = 'Survival'
  pred = predict_scalar(grid_scalar_ge_res_1, y, ytype, G, E, GE = None)
  grid_scalar_ge_res_2 = grid_scalar_ge(scalar_binary['y'], scalar_binary['G'], scalar_binary['E'], ytype='Binary',
                                        num_hidden_layers=1, nodes_hidden_layer=[50], num_epochs=50,
                                        learning_rate1=[0.04, 0.05, 0.06], learning_rate2=[0.035,0.045],
                                        lambda1 = None, lambda2 = [0.08, 0.09], Lambda = [0.1],
                                        threshold = 0.01, model = None, split_type = 1, ratio = [7, 2, 1], 
                                        important_feature = True, plot = False, model_reg = None, isfunc = False)
  grid_scalar_ge_res_3 = grid_scalar_ge(scalar_survival['y'], scalar_survival['G'], scalar_survival['E'], ytype='Survival',
                                        num_hidden_layers=2, nodes_hidden_layer=[1000,100], num_epochs=50,
                                        learning_rate1=[0.02, 0.04, 0.06], learning_rate2=[0.035,0.045],
                                        lambda1 = None, lambda2 = [0.05, 0.09], Lambda = [0.1],
                                        threshold = 0.01, model = None, split_type = 1, ratio = [7, 2, 1], 
                                        important_feature = False, plot = False, model_reg = None, isfunc = False)
  assert grid_scalar_ge_res_1 is not None
  assert grid_scalar_ge_res_2 is not None
  assert grid_scalar_ge_res_3 is not None
  assert pred is not None
