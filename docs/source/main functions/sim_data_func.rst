sim_data_func
===========================

sim_data_scalar
=========================

.. _simdatascalar-label:

Description
------------

This function is used to generate the example data for functions **func_ge** and **grid_func_ge**.
Users can customize the outcomes using the parameter shown in the parameter table below.

See also at :ref:`func_ge <funcge-label>` and :ref:`grid_scalar_ge <gridfuncge-label>`.

Usage
------

.. code-block:: python

    sim_data_func(n, m, ytype, seed = 0)

Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to customize the simulation data.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Description
   * - **n**
     - numeric, sample size.
   * - **m**
     - numeric, the sequence length of each sample.
   * - **ytype**
     - character, "Survival", "Binary" or "Continuous" type of the output y. If not specified, the default is continuous.
   * - **seed**
     - numeric, random seeds each time when data is generated.

Value
-------

The function **sim_data_func** outputs a dictionary including response variable y, scalar variable z and sequence (genotypes) data X..

- **x**: a matrix representing the sequence data, with the number of rows equal to the number of samples.

- **y**: an array representing the response variables.

- **z**: a matrix representing the scalar covariates, with the number of rows equal to the number of samples.

- **location**: a list defining the sampling sites of the sequence (genotypes) data.



Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

    from GENetLib.sim_data_func import sim_data_func
    func_continuous = sim_data_func(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
    x = func_continuous['X']
    y = func_continuous['y']
    z = func_continuous['z']
    location = func_continuous['location']