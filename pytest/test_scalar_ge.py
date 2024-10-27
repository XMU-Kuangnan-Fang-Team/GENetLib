from GENetLib.sim_data_scalar import sim_data_scalar
from GENetLib.scalar_ge import scalar_ge
import matplotlib
matplotlib.use('Agg')

def test_func_ge():
  scalar_continuous = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Continuous', n_inter=30)
  scalar_binary = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Binary', n_inter=10)
  scalar_survival = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Survival', n_inter=10, linear = False)
  scalar_ge_res_1 = scalar_ge(data=scalar_continuous[0], ytype='Continuous', dim_G=500, dim_E=5, haveGE=True, 
                              num_hidden_layers=2, nodes_hidden_layer=[1000,100], 
                              Learning_Rate2=0.035, L2=0.01, Learning_Rate1=0.06, L=0.09, 
                              Num_Epochs=50, t = 0.01, model = None, split_type = 0, ratio = [7, 3], 
                              important_feature = True, plot = True, model_reg = None, isfunc = False)
  scalar_ge_res_2 = scalar_ge(data=scalar_binary[0], ytype='Binary', dim_G=500, dim_E=5, haveGE=True, 
                              num_hidden_layers=2, nodes_hidden_layer=[1000,100], 
                              Learning_Rate2=0.035, L2=0.01, Learning_Rate1=0.06, L=0.09, 
                              Num_Epochs=50, t = 0.01, model = None, split_type = 1, ratio = [7, 2, 1], 
                              important_feature = True, plot = False, model_reg = None, isfunc = False)
  scalar_ge_res_3 = scalar_ge(data=scalar_survival[0], ytype='Survival', dim_G=500, dim_E=5, haveGE=False, 
                              num_hidden_layers=2, nodes_hidden_layer=[1000,100], 
                              Learning_Rate2=0.035, L2=0.01, Learning_Rate1=0.06, L=0.09, 
                              Num_Epochs=50, t = 0.01, model = None, split_type = 1, ratio = [7, 2, 1], 
                              important_feature = True, plot = False, model_reg = None, isfunc = False)
  assert scalar_ge_res_1 is not None
  assert scalar_ge_res_2 is not None
  assert scalar_ge_res_3 is not None

