from GENetLib.sim_data_scalar import sim_data_scalar
from GENetLib.pre_data2 import pre_data2

scalar_continuous = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Continuous', n_inter=30, seed = 111)
pre1 = pre_data2(scalar_continuous['y'], scalar_continuous['G'], scalar_continuous['E'], interaction = scalar_continuous['GE'], ytype = 'Continuous', split_type = 0, ratio = [7, 3])

scalar_binary = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=500, dim_E_Sparse=2, ytype='Binary', n_inter=30, seed = 111)
pre2 = pre_data2(scalar_binary['y'], scalar_binary['G'], scalar_binary['E'], interaction = scalar_binary['GE'], ytype = 'Binary', split_type = 0, ratio = [7, 3])

assert pre1 is not None
assert pre2 is not None