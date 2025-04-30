from GENetLib.sim_data import sim_data_scalar

def test_sim_data_scalar():
    scalar_survival_linear = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=1500, dim_E_Sparse=2, ytype='Survival', n_inter=30)
    assert scalar_survival_linear['y'] is not None
    assert scalar_survival_linear['G'] is not None
    assert scalar_survival_linear['E'] is not None
    assert scalar_survival_linear['GE'] is not None
    assert scalar_survival_linear['interpos'] is not None
