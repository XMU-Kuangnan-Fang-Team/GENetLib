sim_data_func
===========================

.. _simdatafunc-label:

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

The function **sim_data_func** outputs a dictionary including response variable y, scalar variable z and sequence (genotypes) data X.

- **y**: An array The response variable. When the type of output data is "survival", output y is an n*2 array that consists:

1. The minimum of the survival time and censoring time.

2. The event indicator.

- **X**: A matrix representing the sequence data, with the number of rows equal to the number of samples.

- **location**: A list defining the sampling sites of the sequence (genotypes) data.

- **Z**: A matrix representing the scalar covariates, with the number of rows equal to the number of samples.

Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

    from GENetLib.sim_data_func import sim_data_func
    func_continuous = sim_data_func(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
    X = func_continuous['X']
    y = func_continuous['y']
    Z = func_continuous['Z']
    location = func_continuous['location']
