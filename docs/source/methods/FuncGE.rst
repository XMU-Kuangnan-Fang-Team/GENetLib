The FuncGE Model
=========================

.. _funcgemodel-label:

The FuncGE model is utilized to capture G-E interaction effects between functional genetic variables and scalar environmental variables.


Notation 1
-----------

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


Notation 2
-------------

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Notation
   * - Matrix of basis coefficients for the functional genetic variables
     - :math:`\boldsymbol{U}=(\boldsymbol{U}_1, \ldots, \boldsymbol{U}_n)^{\top}` where :math:`\boldsymbol{U}_i=\left(U_{i1}, \ldots, U_{iL_\beta} \right)^{\top}` is computed as the integral :math:`U_{ij} = \int_0^T X_i(t)\psi_j(t)dt`
   * - G-E interaction variables
     - :math:`\boldsymbol{Q}=(\boldsymbol{Q}_1, \ldots, \boldsymbol{Q}_n)^\top` where :math:`\boldsymbol{Q}_i = \boldsymbol{Z}_i \otimes \boldsymbol{U}_i`
   * - Combination of G and G-E variables
     - :math:`\tilde{\boldsymbol{H}} = (\boldsymbol{U}, \boldsymbol{Q})`


Thus, we define :math:`\tilde{\boldsymbol{W}} = \left(\boldsymbol{U}, \boldsymbol{Q}, \boldsymbol{Z} \right)`
where :math:`\tilde{\boldsymbol{W}}_i` represents the variable set for the :math:`i`-th individual.


Loss Functions
-----------------

With the definitions of variable set :math:`\tilde{\boldsymbol{W}}` in place,
the loss function for continuous outcomes can be derived in a form similar to that described in :ref:`ScalarGE <scalargemodel-label>`.


1. Continuous layer:

.. math::

    \tilde{l}_{\text{continuous}}(\boldsymbol{\theta})=\frac1n\sum_{i=1}^n[y_i-g(\tilde{\boldsymbol{W}}_i\boldsymbol{\theta})]^2,

where :math:`\boldsymbol{\theta}` represents the neural network weights.

2. Binary layer:

.. math::

    \tilde{l}_{\text{binary}}(\boldsymbol{\theta}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i\log(p_i) + (1 - y_i) \log (1 - p_i)-\tilde{\boldsymbol{W}}_i \boldsymbol{\theta} \right],

where :math:`p_i = (1 + e^{-\tilde{\boldsymbol{W}}_i\boldsymbol{\theta}})^{-1}`.

3. Cox layer:

.. math::

    \tilde{l}_{\text{Cox}}(\boldsymbol{\theta})=-\sum_{i:\delta_{i}=1 }\biggl[h_{\boldsymbol{\theta}}(\tilde{\boldsymbol{W}}_i)-\log{\sum_{j\in R(T_{i})}e^{h_{\boldsymbol{\theta}}(\tilde{\boldsymbol{W}}_j)}}\biggr],

where :math:`y_i` and :math:`c_i` represent the survival time and censoring time,
:math:`T_i` is the minimum value of :math:`y_i` and :math:`c_i`, :math:`\delta_i` is an indicator variable that :math:`I(y_i \leq c_i)`.
The at-risk set at time :math:`T_{i}` is :math:`R(T_i) = \{i' : T_{i'} \geq T_i\}`, and :math:`h_{\boldsymbol{\theta}}` is the prognostic index.


Penalty Approximation
-------------------------

Define :math:`\tilde{\gamma}_{j}` as the main effect and :math:`\tilde{\beta}_{1j}, \ldots, \tilde{\beta}_{qj}` as the interaction effects.
Then, the coefficient vector related to the corresponding G and G-E variables in :math:`\tilde{\boldsymbol{H}}` is denoted by :math:`\tilde{\boldsymbol{b}}_j = (\tilde{\gamma}_{j}, \tilde{\beta}_{1j}, \ldots, \tilde{\beta}_{qj})^{\top} = (\tilde{b}_{j1}, \ldots, \tilde{b}_{j(q+1)})^{\top}`, where :math:`j = 1, \ldots, p`.

Let :math:`K` represent the total number of fully connected layers within the network architecture, with :math:`\tilde{\omega}_k` indicating the weight matrix allocated to the :math:`k`-th layer.
The objective function can then be formulated as:

.. math::

    \tilde{L}(\boldsymbol{\theta}) = \tilde{l}_{\cdot}(\boldsymbol{\theta}) + \sum_{j=1}^p\rho(|| \tilde{\boldsymbol{b}}_j||;\lambda_1,s) + \sum_{j=1}^p \sum_{k=1}^q \rho(|\tilde{\beta}_{kj}|; \lambda_2, s) + \lambda \biggl( \sum_{k=1}^K||\tilde{\omega}_k||_F^2 \biggr),


where :math:`\tilde{l}_{\cdot}(\boldsymbol{\theta})` is the corresponding loss function for various types of outcomes. 

Similarly, we utilize the same method as :ref:`ScalarGE <scalargemodel-label>` to approximate penalties.
We minimize the objective function, iterating the estimation process until the training converges and parameters stabilize folloing the algorithm below.


Algorithm: Training of FuncGE
-------------------------------

**Input**:

- Functional genetic variables :math:`\boldsymbol{X}(t)` or discrete sequence :math:`\check{\boldsymbol{X}}`;

- Environmental variables :math:`\boldsymbol{Z}`;

- Survival output :math:`(T,\delta)` or continuous outputs :math:`y` or binary output :math:`y`;

- Learning rates of the sparse layer and the fully connected layers :math:`\{\alpha_1,\alpha_2\}`;

- Tuning and regularization parameters of the MCP penalty :math:`\{\lambda_1, \lambda_2, s\}`;

- Tuning parameter of the fully connected layers :math:`\lambda`.

**Data pre-processing**:

- For functional genetic input :math:`\boldsymbol{X}`, format :math:`\tilde{W} = (\boldsymbol{U}, \boldsymbol{Q}, \boldsymbol{Z})`;

- For sequence genetic input :math:`\check{\boldsymbol{X}}`, format :math:`\tilde{\boldsymbol{W}} = (\tilde{\boldsymbol{U}}, \tilde{\boldsymbol{Q}}, \boldsymbol{Z})`.

**Initialize**:

- Sparse layer :math:`\tilde{\boldsymbol{b}}^{(0)}`, :math:`k`-th fully connected layer :math:`\tilde{\omega}_k^{(0)}`, :math:`m = 0`.

**Repeat**:

- Update the approximated MCP penalties with the current estimate :math:`\tilde{\boldsymbol{b}}^{(m)}`;

- Update Loss :math:`= \tilde{l}(\boldsymbol{\theta}) + \text{approximated MCP penalties} + \lambda \sum_{k=1}^{K} \|\tilde{\omega}_{k}^{(m)}\|_F^2`;

- Conduct back propagation, and obtain the gradients :math:`\frac{\partial \text{Loss}}{\partial \tilde{\boldsymbol{b}}_j^{(m)}}` and :math:`\frac{\partial \text{Loss}}{\partial \tilde{\omega}_k^{(m)}}`;

- For :math:`j = 1` to :math:`p` do

  - Update estimates :math:`\tilde{\gamma}_j^{(m+1)} = \tilde{\gamma}_j^{(m)} - \alpha_1 \frac{\partial \text{Loss}}{\partial \tilde{\gamma}_j^{(m)}}`;

  - For :math:`k = 1` to :math:`q` do

    - Update estimates :math:`\tilde{\beta}_{kj}^{(m+1)} = \tilde{\beta}_{kj}^{(m)} - \alpha_1 \frac{\partial \text{Loss}}{\partial \tilde{\beta}_{kj}^{(m)}}`;

  - End for;

- End for;

- For :math:`k = 1` to :math:`K` do

  - Update :math:`\tilde{\omega}_k^{(m+1)} = \tilde{\omega}_k^{(m)} - \alpha_2 \frac{\partial \text{Loss}}{\partial \tilde{\omega}_k^{(m)}}`;

- End for;

- Update :math:`m = m + 1`;

- Until convergence or :math:m reaches its maximum.


Extra: Sequence Data Processing
-------------------------------


We extend our model to accommodate densely sampled discrete data.

For the :math:`i`-th individual, suppose we obtained the densely measured observations :math:`\boldsymbol{\check{X}}_{i} = (X_{i}(t_{i1}), \ldots, X_{i}(t_{m_{i}}) )^{\top}` at different physical positions :math:`\{ t_{i1}, \ldots, t_{im_i} \}`.
Here, :math:`\boldsymbol{\check{X}}_i` is considered a discrete realization of a smooth genetic function :math:`X_i(t)`, where :math:`t \in [0, T]`.

Using functional data analysis, we employ least squares-based smoothing techniques to estimate the function :math:`X_i(t)`.
The function :math:`X_i(t)` can be approximated as:

.. math::

    \hat{X}_i(t) = \check{\boldsymbol{X}}_i^\top \boldsymbol{\Omega}_i (\boldsymbol{\Omega}_i^\top \boldsymbol{\Omega}_i)^{-1} \boldsymbol{\phi}(t),

where :math:`\boldsymbol{\phi}(t) = (\phi_1(t), \ldots, \phi_{L_X}(t))^\top` is a set of basis functions, such as B-splines, Fourier series, or wavelets.
:math:`\Omega_{i}` is an matrix where :math:`\Omega_{i}` in the :math:`j`-th row and :math:`l`-th column is the value of :math:`\phi_{l}(t_{ij})`.

Then we can use this expansion and re-execute the process of the FuncGE model to obtain the final modeling results.

