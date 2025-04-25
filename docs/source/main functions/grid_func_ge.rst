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

    grid_func_ge(y, X, location, Z, ytype, btype, num_hidden_layers, nodes_hidden_layer, num_epochs, learning_rate1, learning_rate2, nbasis1, params1, lambda1 = None, lambda2 = None, Lambda = None, Bsplines = 20, norder1 = 4, split_type = 0, ratio = [7, 3], plot_res = True, plot_beta = True)


Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to build a optimal FuncGE model with given parameters.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **y**
     - numeric, an array representing the response variables.
   * - **X**
     - numeric or dict, a matrix representing the sequence data with the number of rows equal to the number of samples or a "fd" item which represents the functional data.
   * - **location**
     - list, a list defining the sampling sites of the sequence data.
   * - **Z**
     - numeric, a matrix representing the scalar covariates, with the number of rows equal to the number of samples.
   * - **ytype**
     - character, "Survival", "Binary" or "Continuous" type of the output y.
   * - **btype**
     - character, "Bspline", "Exponential", "Fourier", "Monomial" or "power" type of spline.
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
   * - **nbasis1**
     - integer, an integer specifying the number of basis functions that constitutes the genetic variation function.
   * - **params1**
     - integer, in addition to rangeval1 (a vector of length 2 giving the lower and upper limits of the range of permissible values for the genetic variation function) and nbasis1, all bases have one or two parameters unique to that basis type or shared with one other.
   * - **lambda1**
     - numeric or None, tuning parameter of the first MCP penalization.
   * - **lambda2**
     - list, tuning parameters of the second MCP penalization.
   * - **Lambda**
     - list, tuning parameters of L2 penalization.
   * - **Bsplines**
     - integer, an integer specifying the number of basis functions that constitutes the genetic effect function.
   * - **norder1**
     - integer, an integer specifying the order of bsplines that constitutes the genetic effect function, which is one higher than their degree. The default of 4 gives cubic splines.
   * - **split_type**
     - integer, types of data split. If split_type = 0, the data is divided into a training set and a validation set. If split_type = 1, the data is divided into a training set, a validation set and a test set.
   * - **ratio**
     - list, the ratio of data split.
   * - **plot_res**
     - bool, "True" or "False", whether or not to show the line plot of residuals with the number of neural network epochs.
   * - **plot_beta**
     - bool, "True" or "False", whether or not to show the graph of predicted functions.


Value
-------

The function **grid_func_ge** outputs a tuple including training results and optimal parameters of the FuncGE model.

- Values of tunning parameters after grid search.

- Residual of the training set.

- Residual of the validation set.

- C index (y is survival) or R2 (y is continuous or binary) of the training set.

- C index (y is survival) or R2 (y is continuous or binary) of the validation set.

- A neural network after training.

- Estimated coefficients of the chosen basis functions for the genetic effect function :math:`\beta_0(t)` and interaction items :math:`\beta_k(t)`.

- The estimated genetic effect function :math:`\beta(t)` and interaction items :math:`\beta_k(t)`.

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
    learning_rate2 = [0.008, 0.009, 0.01]
    Lambda = [0.002, 0.003, 0.004, 0.005, 0.006]
    learning_rate1 = [0.02, 0.03, 0.04, 0.05]
    lambda2 = [0.05, 0.06, 0.07, 0.08]
    num_epochs = 100
    nbasis1 = 7
    params1 = 4
    func_continuous = sim_data_func(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
    y = func_continuous['y']
    Z = func_continuous['Z']
    location = func_continuous['location']
    X = func_continuous['X']
    grid_func_ge_res = grid_func_ge(y, X, location, Z, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer, num_epochs, learning_rate1, learning_rate2, nbasis1, params1, lambda1 = None, lambda2 = lambda2, Lambda = Lambda, Bsplines = 15, norder1 = 4, split_type = 1, ratio = [3, 1, 1], plot_res = False, plot_beta = True)

