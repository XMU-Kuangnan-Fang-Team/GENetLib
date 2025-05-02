plot_fd
=========================

.. _plotfunc-label:

*Plot the functional data.*


Description
------------

This function aims to plot the funcational data. The form of functional data is defined by this package. 


Usage
------

.. code-block:: python

    plot_fd(x, y=None, xlab=None, ylab=None)


Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to plot a figure of functional data.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **x**
     - dict, functional data object(s) to be plotted.
   * - **y**
     - str, sequence of points at which to evaluate the functions x and plot on the horizontal axis. 
   * - **xlab**
     - str, a label for the horizontal axis.
   * - **ylab**
     - str, a label for the vertical axis.


Value
-------

The function **plot_fd** plots the figure of functional data.
Here are two example graphs for this function:

.. raw:: html

   <div style="text-align: center;">

.. |image1| image:: /_static/plot_fd_1.png
   :width: 350px

.. |image2| image:: /_static/plot_fd_2.png
   :width: 350px

|image1| |image2|

.. raw:: html

   </div>


Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

  from GENetLib.fda_func import dense_to_func
  # dense_to_func: a function to convert densely measured data into functional data
  from GENetLib.sim_data import sim_data_func
  from GENetLib.plot_gene import plot_fd
  func_continuous = sim_data_func(2, 30, 'Continuous')
  location = list(func_continuous['location'])
  X = func_continuous['X']
  dense_to_func_res = dense_to_func(location, X, btype = "Bspline", nbasis = 5, params = 4, Plot = False)
  plot_fd(dense_to_func_res)
