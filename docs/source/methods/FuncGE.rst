The FuncGE Model
=========================

.. _funcgemodel-label:

The FuncGE model is utilized to capture G-E interaction effects between functional genetic variables and scalar environmental variables.


Notation
---------

Consider a dataset with :math:`n` independent individuals.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Notation
   * - Outcome variables
     - :math:`\boldsymbol{y} = (y_1, \ldots, y_n)^{\top}`
   * - Functional genetic variables
     - :math:`X_i(t)` where :math:`t \in \mathcal{T}`, with :math:`\mathcal{T} = [0,T]`
   * - Scalar environmental variables
     - :math:`\boldsymbol{Z}_i = \left(z_{i1}, \ldots, z_{iq} \right)^{\top}`
   * - Genetic effect function
     - :math:`\beta_0(t)`
   * - Interaction effect functions
     - :math:`\beta_k(t)` for :math:`k = 1, \ldots, q`


Model
---------

For continuous outcomes, we consider the following model from a functional data analysis perspective:

.. math::

    y_i=g\left(\sum_{k=1}^qz_{ik}\gamma_k+\int_0^TX_i(t)\beta_0(t)dt+\sum_{k=1}^qz_{ik}\int_0^TX_i(t)\beta_k(t)dt\right)+\varepsilon_i

where :math:`\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_q)^{\top}` is the :math:`q`-dimensional vector of coefficients.
The error term :math:`\epsilon_i` follows a normal distribution :math:`N(0, \sigma_e^2)` and is assumed to be independent of both :math:`\boldsymbol{Z}_i` and :math:`X_i(t)`.
:math:`g` is the functional form.


Spline Regression
---------------------

Based on spline regression techniques, we approximate :math:`\beta_k(t)` for :math:`k = 0, 1, \ldots, q` using a set of B-spline basis functions :math:`\boldsymbol{\psi}(t) = \left( \psi_1(t), \ldots, \psi_{L_\beta}(t) \right)^\top`.

These basis functions are：

- Knots: :math:`M_n` knots that are uniformly distributed.
- Interval: :math:`\mathcal{T}`.
- Degree :math:`d`.
- :math:`L_\beta = M_n + d`.

Thus, :math:`\beta_k(t)` can be approximated as:

.. math::

    \beta_k(t)=\boldsymbol{\psi}(t)^\top \boldsymbol{\theta}_k, \; k=0,1,\ldots,q

where :math:`\boldsymbol{\theta}_{k} = (\theta_{k1}, \ldots, \theta_{kL_\beta})^{\top}` are the coefficients for the B-spline basis functions.

Using this approximation, we can rewrite the model above:

.. math::

    y_{i} = g\left\{\sum_{k=1}^q z_{ik} \gamma_k + \left[\int_0^T X_i(t) \boldsymbol{\psi}(t)^\top \mathrm{d}t\right] \boldsymbol{\theta_0} + \sum_{k=1}^q z_{ik} \left[\left(\int_0^T X_i(t) \boldsymbol{\psi}(t)^\top \mathrm{d}t\right) \boldsymbol{\theta_k}\right]\right\} + \varepsilon_i

    