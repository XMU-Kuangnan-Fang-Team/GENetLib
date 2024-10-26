import GENetLib
from GENetLib.sim_data_scalar import sim_data_scalar
scalar_survival_linear = sim_data_scalar(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500, dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)
scalar_survival_linear_data = scalar_survival_linear[0]
scalar_survival_linear_inter = scalar_survival_linear[1]


from GENetLib.sim_data_func import sim_data_func
func_continuous = sim_data_func(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
x = func_continuous['X']
y = func_continuous['y']
z = func_continuous['z']
location = func_continuous['location']


from GENetLib.sim_data_func import sim_data_func
from GENetLib.func_ge import func_ge
num_hidden_layers = 2
nodes_hidden_layer = [100,10]
Learning_Rate2 = 0.035
L2 = 0.01
Learning_Rate1 = 0.02
L = 0.01
Num_Epochs = 50
nbasis1 = 5
params1 = 4
func_continuous = sim_data_func(n = 1500, m = 30, ytype = 'Continuous', seed = 123)
y = func_continuous['y']
z = func_continuous['z']
location = func_continuous['location']
X = func_continuous['X']
func_ge_res = func_ge(y, z, location, X, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, Bsplines = 5, norder1 = 4, model = None, split_type = 1, ratio = [3, 1, 1], plot_res = True)


from GENetLib.sim_data_func import sim_data_func
from GENetLib.grid_func_ge import grid_func_ge
num_hidden_layers = 2
nodes_hidden_layer = [100, 10]
Learning_Rate2 = [0.008, 0.009, 0.01]
L2 = [0.002, 0.003, 0.004, 0.005, 0.006]
Learning_Rate1 = [0.02, 0.03, 0.04, 0.05]
L = [0.05, 0.06, 0.07, 0.08]
Num_Epochs = 100
nbasis1 = 7
params1 = 4
func_continuous = sim_data_func(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
y = func_continuous['y']
z = func_continuous['z']
location = func_continuous['location']
X = func_continuous['X']
grid_func_ge_res = grid_func_ge(y, z, location, X, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, Bsplines = 15, norder1 = 4, model = None, split_type = 0, ratio = [7,3], plot_res = True)

