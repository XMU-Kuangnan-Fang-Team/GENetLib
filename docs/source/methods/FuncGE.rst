The FuncGE Model
=========================

.. _funcgemodel-label:

The FuncGE model is utilized to capture gene-environment interaction effects between functional genetic variables and scalar environmental variables.

Similar to the ScalarGE model, FuncGE models the genetic effects and G-E interaction effects through a similar neural network architecture. For :math:`n` independent individuals, let the outcome variables be :math:`\boldsymbol{y} = (y_1, \ldots, y_n)^{\top}`. Denote :math:`X_i(t)` as the functional genetic variables, where :math:`t \in \mathcal{T}`, with :math:`\mathcal{T} = [0,T]` serving as an illustrative interval. Let :math:`\boldsymbol{Z}_i = \left(z_{i1}, \ldots, z_{iq} \right)^{\top}` represent the scalar environmental variables. For continuous outcomes, we consider the following model from a functional data analysis perspective:

.. math::

    y_i=g\left(\sum_{k=1}^qz_{ik}\gamma_k+\int_0^TX_i(t)\beta_0(t)dt+\sum_{k=1}^qz_{ik}\int_0^TX_i(t)\beta_k(t)dt\right)+\varepsilon_i,
    :label: equ: model_FuncGE

where :math:`\boldsymbol{\gamma} = (\gamma_1, \ldots, \gamma_q)^{\top}` is the :math:`q`-dimensional vector of coefficients. The functions :math:`\beta_0(t)` and :math:`\beta_k(t)`, for :math:`k = 1, \ldots, q`, represent the genetic effect function and interaction effect functions respectively. The error term :math:`\epsilon_i` follows a normal distribution :math:`N(0, \sigma_e^2)` and is assumed to be independent of both :math:`\boldsymbol{Z}_i` and :math:`X_i(t)`. The functional form :math:`g` leverages the neural network's ability to capture nonlinear relationships.

Based on spline regression techniques, we approximate :math:`\beta_k(t)` for :math:`k = 0, 1, \ldots, q` using a set of B-spline basis functions :math:`\boldsymbol{\psi}(t) = \left( \psi_1(t), \ldots, \psi_{L_\beta}(t) \right)^\top`. These basis functions are defined over :math:`M_n` knots that are uniformly distributed over the interval :math:`\mathcal{T}`, with degree :math:`d`, resulting in :math:`L_\beta = M_n + d`. Thus, :math:`\beta_k(t)` can be approximated as:

.. math::

    \beta_k(t)=\boldsymbol{\psi}(t)^\top \boldsymbol{\theta}_k, \; k=0,1,\ldots,q,

where :math:`\boldsymbol{\theta}_{k} = (\theta_{k1}, \ldots, \theta_{kL_\beta})^{\top}` are the coefficients for the B-spline basis functions. Using this approximation, we can rewrite the model (:eq:`equ: model_FuncGE`) as:

.. math::

    y_{i}&=g\biggl\{\sum_{k=1}^qz_{ik}\gamma_k+\left[\int_0^TX_i(t)\boldsymbol{\psi}(t)^\top\mathrm{d}t\right]\boldsymbol{\theta_0}+\sum_{k=1}^qz_{ik}\left[\left(\int_0^TX_i(t)\boldsymbol{\psi}(t)^\top\mathrm{d}t\right)\boldsymbol{\theta_k}\right]\biggr\}+\varepsilon_i.
