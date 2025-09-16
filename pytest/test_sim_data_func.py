from GENetLib.sim_data import sim_data_func


def test_sim_data_func():
    func_continuous = sim_data_func(n=1500, m=30, ytype="Continuous", seed=123)
    assert func_continuous["X"] is not None
    assert func_continuous["y"] is not None
    assert func_continuous["Z"] is not None
    assert func_continuous["location"] is not None
