predict_func
=========================

.. _predictfunc-label:

*Making prediction for G-E interaction analysis via deep learning when the input X is functional data.*

Description
------------

This function provides a predict function for the result of :ref:`FuncGE <funcgemodel-label>` model, including discrete observations and functional inputs.

See also at :ref:`func_ge <funcge-label>` and :ref:`grid_func_ge <gridfuncge-label>`.

Usage
------

.. code-block:: python

    predict_func(ge_res, y, ytype, G, E, location, nbasis = 15, params = 4, nbasis1 = 15, params1 = 4, Bsplines = 20, norder1 = 4)

Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to build a customizable FuncGE model.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **ge_res**
     - tuple, contains the trained G-E network results.
   * - **y**
     - numeric, an array representing the response variables.
   * - **ytype**
     - character, "Survival", "Binary" or "Continuous" type of the output y.
   * - **G**
     - numeric or list, a matrix representing the sequence data with the number of rows equal to the number of samples or a list of "fd" items which represents the functional data.
   * - **E**
     - numeric, a matrix representing the scalar covariates, with the number of rows equal to the number of samples.
   * - **location**
     - list, a list defining the sampling sites of the sequence data.
   * - **nbasis**
     - integer, an integer specifying the number of basis functions that constitutes the genetic variation function (discrete observations data).
   * - **params**
     - integer, in addition to rangeval1 (a vector of length 2 giving the lower and upper limits of the range of permissible values for the genetic variation function) and nbasis1, all bases have one or two parameters unique to that basis type or shared with one other (discrete observations data).
   * - **nbasis1**
     - integer, an integer specifying the number of basis functions that constitutes the genetic variation function (discrete observations data).
   * - **params1**
     - integer, in addition to rangeval1 (a vector of length 2 giving the lower and upper limits of the range of permissible values for the genetic variation function) and nbasis1, all bases have one or two parameters unique to that basis type or shared with one other (discrete observations data).
   * - **Bsplines**
     - integer, an integer specifying the number of basis functions that constitutes the genetic effect function (functional inputs).
   * - **norder1**
     - integer, an integer specifying the order of bsplines that constitutes the genetic effect function, which is one higher than their degree. The default of 4 gives cubic splines (functional inputs).

Value
-------

The function **predict_func** outputs a tensor including prediction results of the FuncGE model. The length of the tensor equals to the number of observations.


Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

  from GENetLib.sim_data import sim_data_func
  from GENetLib.grid_func_ge import grid_func_ge
  from GENetLib.predict_ge import predict_func
  ytype = 'Continuous'
  num_hidden_layers = 2
  nodes_hidden_layer = [100, 10]
  learning_rate2 = [0.008, 0.009, 0.01]
  Lambda = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
  learning_rate1 = [0.02, 0.03, 0.04, 0.05]
  lambda2 = [0.05, 0.06, 0.07, 0.08]
  num_epochs = 100
  nbasis1 = 7
  params1 = 4
  func_continuous = sim_data_func(n = 1000, m = 100, ytype = ytype, seed = 1)
  y = func_continuous['y']
  Z = func_continuous['Z']
  location = func_continuous['location']
  X = func_continuous['X']
  grid_func_ge_res = grid_func_ge(y, X, location, Z, ytype, 'Bspline', num_hidden_layers, nodes_hidden_layer,
                                  num_epochs, learning_rate1, learning_rate2, nbasis1, params1, 
                                  lambda1 = None, lambda2 = lambda2, Lambda = Lambda, Bsplines = 15,  
                                  norder1 = 4, model = None, split_type = 1, ratio = [3, 1, 1], plot_res = False,
                                  plot_beta = True)
  pred = predict_func(grid_func_ge_res, y, ytype, X, Z, location)
