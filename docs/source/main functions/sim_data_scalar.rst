sim_data_scalar
=========================

.. _simdatascalar-label:

Description
------------

This function is used to generate the example data for functions **scalar_ge** and **grid_scalar_ge**.
Users can customize the outcomes using the parameter shown in the parameter table below.

See also at :ref:`**scalar_ge** <scalarge-label>` and :ref:`**grid_scalar_ge** <gridscalarge-label>`.

Usage
------

.. code-block:: python

   sim_data_scalar(rho_G, rho_E, dim_G, dim_E, n, dim_E_Sparse=0, ytype='Survival', n_inter=None, linear=True, seed=0)

Parameters
----------

This part shows the meanings and data types of parameters. Users can check the table below to customize the simulation data.

.. list-table:: 
   :widths: 20 10 70
   :header-rows: 1
   :align: center

   * - Parameter
     - | 
     - Description
   * - rho_G
     - | 
     - numeric, correlation of gene variables.
   * - rho_E
     - | 
     - numeric, correlation of environment variables.
   * - dim_G
     - | 
     - numeric, dimension of gene variables.
   * - dim_E
     - | 
     - numeric, dimension of environment variables.
   * - n
     - | 
     - numeric, sample size.
   * - dim_E_Sparse
     - | 
     - numeric, dimension of sparse environment variables.
   * - ytype
     - | 
     - character, "Survival", "Binary" or "Continuous" type of the output y. If not specified, the default is survival.
   * - n_inter
     - | 
     - numeric, number of interaction effect variables.
   * - linear
     - | 
     - bool, "True" or "False", whether or not to generate linear data. The default is True.
   * - seed
     - | 
     - numeric, random seeds each time when data is generated.

Value
-------

The function **sim_data_scalar** outputs a tuple including generated data and the positions of interaction effect variables.

**data**: A dataframe contains gene variables, environment variables, interaction variables and output y.
When the type of output data is "survival", output y is an n*2 array that consists:

- The minimum of the survival time and censoring time.

- The event indicator.

**interaction effect variables**: An array contains the positions of interaction effect variables.



Examples
-------------

Here is a quick example for using this function:

.. code-block:: python

   import GENetLib
   from GENetLib.sim_data_scalar import sim_data_scalar
   scalar_survival_linear = sim_data_scalar(rho_G=0.25, rho_E=0.3, dim_G=500, dim_E=5, n=1500, dim_E_Sparse=2, ytype='Survival', n_inter=30)
   scalar_survival_linear_data = scalar_survival_linear['data']
   scalar_survival_linear_inter = scalar_survival_linear['interpos']