from GENetLib.sim_data_scalar import sim_data_scalar

def test_sim_data_scalar():
    scalar_survival_linear = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=1500, dim_E_Sparse=2, ytype='Survival', n_inter=30)
    assert scalar_survival_linear[0] is not None
    assert scalar_survival_linear[1] is not None
test_sim_data_scalar
