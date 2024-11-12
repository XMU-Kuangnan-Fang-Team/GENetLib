grid_func_ge
=========================

.. _gridfuncge-label:

*Grid search for func_ge.*


Description
------------

This function performs grid search for func_ge over a grid of values for the regularization parameter `L`, `L2` and learning rate `Learning_Rate1`, `Learning_Rate2`.

See also at :ref:`sim_data_func <simdatafunc-label>` and :ref:`func_ge <funcge-label>`. The model is :ref:`FuncGE <funcgemodel-label>`.


Usage
------

.. code-block:: python

    grid_func_ge(y, z, location, X, ytype, btype, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, t = None, Bsplines = 20, norder1 = 4, model = None, split_type = 0, ratio = [7, 3], plot_res = True, plot_beta = True)


Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to build a optimal FuncGE model with given parameters.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center



Value
-------

The function **grid_func_ge** outputs a tuple including training results and optimal parameters of the FuncGE model.

- Values of tunning parameters after grid search.

- Residual of the training set.

- Residual of the validation set.

- C index (y is survival) or R2 (y is continuous or binary) of the training set.

- C index (y is survival) or R2 (y is continuous or binary) of the validation set.

- A neural network after training.

- Estimated coefficients of the chosen basis functions for the genetic effect function beta0(t) and interaction items betak(t).

- The estimated genetic effect function beta(t) and interaction items betak(t).

Here is an example output for an established model:

.. image:: /_static/grid_func_ge.png
   :width: 700
   :align: center

In terms of visualization, this function can output the plots of reconstructed functions. Here is an example output:

   
.. raw:: html

   <div style="text-align: center;">

.. |image1| image:: /_static/grid_func_ge_0.png
   :width: 250px

.. |image2| image:: /_static/grid_func_ge_1.png
   :width: 250px

.. |image3| image:: /_static/grid_func_ge_2.png
   :width: 250px

|image1| |image2| |image3|

.. raw:: html

   </div>



Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

    from GENetLib.sim_data_func import sim_data_func
    from GENetLib.grid_func_ge import grid_func_ge
    num_hidden_layers = 2
    nodes_hidden_layer = [100, 10]
    Learning_Rate2 = [0.005, 0.01, 0.015]
    L2 = [0.005, 0.01, 0.015]
    Learning_Rate1 = [0.001, 0.005]
    L = [0.005, 0.006, 0.007]
    Num_Epochs = 50
    nbasis1 = 5
    params1 = 4
    func_continuous = sim_data_func(n = 1000, m = 30, ytype = 'Continuous', seed = 1)
    y = func_continuous['y']
    z = func_continuous['z']
    location = func_continuous['location']
    X = func_continuous['X']
    grid_func_ge_res = grid_func_ge(y, z, location, X, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, Bsplines = 5, norder1 = 4, model = None, split_type = 0, ratio = [7,3], plot_res = True)
