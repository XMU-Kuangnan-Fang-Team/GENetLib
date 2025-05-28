predict_scalar
=========================

.. _predictscalar-label:

*Making prediction for G-E interaction analysis via deep learning when the input X is scalar data.*

Description
------------

This function provides a predict function for the result of :ref:`ScalarGE <scalargemodel-label>` model.

See also at :ref:`scalar_ge <scalarge-label>` and :ref:`grid_scalar_ge <gridscalarge-label>`.

Usage
------

.. code-block:: python

    predict_scalar(ge_res, y, ytype, G, E, GE = None)

Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to build a customizable ScalarGE model.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **ge_res**
     - tuple, contains the trained G-E network results.
   * - **y**
     - array or dataframe, the response variable.
   * - **ytype**
     - character, "Survival", "Binary" or "Continuous" type of the output y.
   * - **G**
     - array or dataframe, the scalar genetic variable.
   * - **E**
     - array or dataframe, the scalar environmental variable.
   * - **GE**
     - Nonetype or array or dataframe, the G-E variable. If GE = None, the function will calculate G-E terms automatically.


Value
-------

The function **predict_scalar** outputs a tensor including prediction results of the ScalarGE model. The length of the tensor equals to the number of observations.


Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

  from GENetLib.sim_data import sim_data_scalar
  from GENetLib.grid_scalar_ge import grid_scalar_ge
  from GENetLib.predict_ge import predict_scalar
  ytype = 'Survival'
  num_hidden_layers = 2
  nodes_hidden_layer = [1000, 100]
  learning_rate2 = [0.035, 0.045]
  Lambda = [0.1]
  learning_rate1 = [0.02, 0.06]
  lambda2 = [0.05, 0.09]
  num_epochs = 100
  scalar_survival_linear = sim_data_scalar(rho_G = 0.75, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 500, dim_E_Sparse = 2, ytype = ytype, n_inter = 30)
  y = scalar_survival_linear['y']
  G = scalar_survival_linear['G']
  E = scalar_survival_linear['E']
  grid_scalar_ge_res = grid_scalar_ge(y, G, E, ytype, num_hidden_layers, nodes_hidden_layer, num_epochs,
                                      learning_rate1, learning_rate2, lambda1 = None, lambda2 = lambda2,
                                      Lambda = Lambda, threshold = 0.01)
  pred = predict_scalar(grid_scalar_ge_res, y, ytype, G, E, GE = None)
