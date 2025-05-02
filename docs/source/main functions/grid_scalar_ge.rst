grid_scalar_ge
=========================

.. _gridscalarge-label:

*Grid search for scalar_ge.*


Description
------------

This function performs grid search for **scalar_ge** over a grid of values for the regularization parameter `L`, `L2`` and learning rate `Learning_Rate1`, `Learning_Rate2`.

See also at :ref:`sim_data_scalar <simdatascalar-label>` and :ref:`scalar_ge <scalarge-label>`. The model is :ref:`ScalarGE <scalargemodel-label>`.


Usage
------

.. code-block:: python

    grid_scalar_ge(y, G, E, ytype, num_hidden_layers, nodes_hidden_layer, num_epochs, learning_rate1, learning_rate2, lambda1 = None, lambda2 = None, Lambda = None, threshold = None, split_type = 0, ratio = [7, 3], important_feature = True, plot = True)


Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to build a optimal ScalarGE model with given parameters.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **y**
     - array or dataframe, the response variable.
   * - **G**
     - array or dataframe, the scalar genetic variable.
   * - **E**
     - array or dataframe, the scalar environmental variable.
   * - **ytype**
     - character, "Survival", "Binary" or "Continuous" type of the output y.
   * - **num_hidden_layers**
     - numeric, number of hidden layers in the neural network.
   * - **nodes_hidden_layer**
     - list, contains number of nodes in each hidden layer.
   * - **num_epochs**
     - numeric, number of epochs for neural network training.
   * - **learning_rate1**
     - list, learning rates of sparse layers.
   * - **learning_rate2**
     - list, learning rates of hidden layers.
   * - **lambda1**
     - numeric or None, tuning parameter of the first MCP penalization.
   * - **lambda2**
     - list, tuning parameters of the second MCP penalization.
   * - **Lambda**
     - list, tuning parameters of L2 penalization.
   * - **threshold**
     - numeric, threshold in the selection of important features.
   * - **split_type**
     - integer, types of data split. If split_type = 0, the data is divided into a training set and a validation set. If split_type = 1, the data is divided into a training set, a validation set and a test set.
   * - **ratio**
     - list, the ratio of data split.
   * - **important_feature**
     - bool, "True" or "False", whether or not to show output features.
   * - **plot**
     - bool, "True" or "False", whether or not to show the line plot of residuals with the number of neural network epochs.


Value
-------

The function **grid_scalar_ge** outputs a tuple including training results and optimal parameters of the ScalarGE model.

- Values of tunning parameters after grid search.

- Residual of the training set.

- Residual of the validation set.

- C index (y is survival) or R2 (y is continuous or binary) of the training set.

- C index (y is survival) or R2 (y is continuous or binary) of the validation set.

- A neural network after training.

- Important features of gene variables.

- Important features of GE interaction variables.

Here is an example output for an established model:

.. image:: /_static/grid_scalar_ge.png
   :width: 700
   :align: center

In terms of visualization, this function can output the line plot of residuals with the number of neural network epochs. Here is an example output:

.. image:: /_static/grid_scalar_ge_train.png
   :width: 500
   :align: center


Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

    from GENetLib.sim_data import sim_data_scalar
    from GENetLib.grid_scalar_ge import grid_scalar_ge
    ytype = 'Survival'
    num_hidden_layers = 2
    nodes_hidden_layer = [1000, 100]
    learning_rate2 = [0.035, 0.045]
    Lambda = [0.1]
    learning_rate1 = [0.01, 0.02, 0.03, 0.04, 0.05]
    lambda2 = [0.04, 0.06, 0.07, 0.09]
    num_epochs = 100
    scalar_survival_linear = sim_data_scalar(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500, dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)
    y = scalar_survival_linear['y']
    G = scalar_survival_linear['G']
    E = scalar_survival_linear['E']
    grid_scalar_ge_res = grid_scalar_ge(y, G, E, ytype, num_hidden_layers, nodes_hidden_layer, num_epochs, learning_rate1, learning_rate2, lambda1 = None, lambda2 = lambda2, Lambda = Lambda, threshold = 0.05)

