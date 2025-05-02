plot_rawdata
=========================

.. _plotrawdata-label:

*Visualization of the sequence (genotypes) data.*


Description
------------

This function aims to plot the sequence data. The horizontal axis represents the sampling sites, and the vertical axis represents the sequence values.


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
     - str, a specification for the default plotting color.
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

.. raw:: html

   <div style="text-align: center;">

.. |image1| image:: /_static/plot_rawdata_1.png
   :width: 350px

.. |image2| image:: /_static/plot_rawdata_2.png
   :width: 350px

|image1| |image2|

.. raw:: html

   </div>


Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

    from GENetLib.sim_data import sim_data_func
    from GENetLib.plot_gene import plot_rawdata
    seed = 123
    func_survival = sim_data_func(3, 30, 'Survival', seed = seed)
    location = list(func_survival['location'])
    X = func_survival['X']
    plot_rawdata(location, X, show_legend = True)
