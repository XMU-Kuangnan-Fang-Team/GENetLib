from GENetLib.sim_data import sim_data_scalar
from GENetLib.scalar_ge import scalar_ge
import matplotlib
matplotlib.use('Agg')

def test_scalar_ge():
  scalar_continuous = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Continuous', n_inter=30, seed = 123)
  scalar_binary = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Binary', n_inter=10, seed = 123)
  scalar_survival = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Survival', n_inter=10, linear = False, seed = 123)
  scalar_ge_res_1 = scalar_ge(scalar_continuous['y'], scalar_continuous['G'], scalar_continuous['E'], ytype='Continuous', 
                              num_hidden_layers=2, nodes_hidden_layer=[1000,100], num_epochs=50, learning_rate1=0.06, learning_rate2=0.035,
                              lambda1 = 0.1, lambda2=0.09, Lambda=0.01, threshold = 0.01, model = None, split_type = 0, ratio = [7, 3],
                              important_feature = True, plot = True, model_reg = None, isfunc = False)
  scalar_ge_res_2 = scalar_ge(scalar_binary['y'], scalar_binary['G'], scalar_binary['E'], ytype='Binary',
                              num_hidden_layers=2, nodes_hidden_layer=[1000,100], num_epochs=50, learning_rate1=0.06, learning_rate2=0.035,
                              lambda1 = 0.1, lambda2=0.09, Lambda=0.01, threshold = 0.01, model = None, split_type = 1, ratio = [7, 2, 1],
                              important_feature = True, plot = False, model_reg = None, isfunc = False)
  scalar_ge_res_3 = scalar_ge(scalar_survival['y'], scalar_survival['G'], scalar_survival['E'], ytype='Survival',
                              num_hidden_layers=2, nodes_hidden_layer=[1000,100], num_epochs=50, learning_rate1=0.06, learning_rate2=0.035,
                              lambda1 = 0.1, lambda2=0.09, Lambda=0.01, threshold = 0.01, model = None, split_type = 1, ratio = [7, 2, 1],
                              important_feature = True, plot = False, model_reg = None, isfunc = False)
  assert scalar_ge_res_1 is not None
  assert scalar_ge_res_2 is not None
  assert scalar_ge_res_3 is not None

