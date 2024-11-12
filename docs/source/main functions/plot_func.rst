plot_func
=========================

.. _plotfunc-label:

*Plot the functional data.*


Description
------------

This function aims to plot the funcational data. The form of functional data is defined by this package. 


Usage
------

.. code-block:: python

    plot_rawdata(location, X, color = None, pch = 4, cex = 0.9, show_legend = True)


Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to plot a customizable sequence figure.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **location**
     - list, defines the sampling sites of the sequence (genotypes) data.
   * - **X**
     - matrix, represents the sequence data with the number of rows equal to the number of samples.
   * - **color**
     - str, Aspecification for the default plotting color.
   * - **pch**
     - float, either an integer specifying a symbol or a single character to be used as the default in plotting points.
   * - **cex**
     - float, a numerical value giving the amount by which plotting text and symbols should be magnified relative to the default. 
   * - **show_legend**
     - bool, whether to show the legend of the plot. When you plot many sequnces, it is recommended to set the option to False.


Value
-------

The function **plot_rawdata** plots the figure of the sequence (genotypes) data.
Here are two example graphs for this function:

.. image:: /_static/plot_rawdata_1.png
   :width: 500
   :align: center

.. image:: /_static/plot_rawdata_2.png
   :width: 500
   :align: center


Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

    from GENetLib.sim_data_func import sim_data_func
    from GENetLib.plot_rawdata import plot_rawdata
    seed = 123
    func_survival = sim_data_func(3, 30, 'Survival', seed = seed)
    location = list(func_survival['location'])
    X = func_survival['X']
    plot_rawdata(location, X, show_legend = True)
