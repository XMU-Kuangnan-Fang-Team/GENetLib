The ScalarGE Model
=========================

.. _scalargemodel-label:

The ScalarGE model is designed to characterize gene-environment (G-E) interaction effects between high-dimensional (scalar) genetic variables and environmental variables.


Notation
----------

Consider a dataset with :math:`n` independent individuals.

.. list-table:: 
   :widths: 30 70
   :header-rows: 1
   :align: center

   * - Parameter
     - Notation
   * - Outcome variables
     - :math:`\boldsymbol{y} = (y_1, \ldots, y_n)^{\top}`
   * - G variables
     - :math:`\boldsymbol{X}=(\boldsymbol{X}_1, \ldots, \boldsymbol{X}_n)^{\top}` where :math:`\boldsymbol{X}_i=(X_{i1}, \dots, X_{ip})`
   * - E variables
     - :math:`\boldsymbol{Z} = (\boldsymbol{Z}_1, \ldots, \boldsymbol{Z}_n)^{\top}` where :math:`\boldsymbol{Z}_i=(Z_{i1}, \dots, Z_{iq})`
   * - G-E interaction variables
     - :math:`\boldsymbol{V} = (\boldsymbol{V}_1, \ldots, \boldsymbol{V}_n)^\top` where :math:`\boldsymbol{V}_i = \boldsymbol{Z}_i \otimes \boldsymbol{X}_i`
   * - Combination of G and G-E variables
     - :math:`\boldsymbol{H} = (\boldsymbol{X}, \boldsymbol{V})`

Thus, we define :math:`\boldsymbol{W}=(\boldsymbol{X}, \boldsymbol{V}, \boldsymbol{Z})` where :math:`\boldsymbol{W}_i` is the variable set for the :math:`i`-th individual.


Loss functions
---------------

We designed three types of output layers for different outcomes:

- Continuous layer, suited for outcomes like disease risk.

- Binary layer, tailored for binary outcomes such as treatment response.

- Cox layer, applicable to survival outcomes such as time to an event.

Next we generate the loss functions for these three layers:

**Continuous layer**

.. math::
    l_{\text{continuous}}(\boldsymbol{\theta})=\frac{1}{n}\sum_{i=1}^n \left[ y_i-g(\boldsymbol{W}_i\boldsymbol{\theta})\right]^2

where :math:`\boldsymbol{\theta}` represents the neural network weights, and :math:`g` is the functional form.

**Binary layer**

.. math::
    l_{\text{binary}}(\boldsymbol{\theta}) = -\frac{1}{n} \sum_{i=1}^n \left[ y_i\log(p_i) + (1 - y_i) \log (1 - p_i)-\boldsymbol{W}_i \boldsymbol{\theta} \right]

where :math:`p_i = (1 + e^{-\boldsymbol{W}_i \boldsymbol{\theta}})^{-1}` is the sigmoid function, and :math:`y_i` represents the binary outcome.

**Cox layer**

.. math::
    l_{\text{Cox}}(\boldsymbol{\theta})=-\sum_{i:\delta_{i}=1}\biggl[h_{\boldsymbol{\theta}}(\boldsymbol{W}_i)-\log{\sum_{j\in R(T_{i})}e^{h_{\boldsymbol{\theta}}(\boldsymbol{W}_j)}}\biggr]

where :math:`y_i` and :math:`c_i` represent the survival time and censoring time, respectively. The minimum of :math:`y_i` and :math:`c_i` is denoted by :math:`T_i`, and the event indicator :math:`\delta_i` is defined as :math:`I(y_i \leq c_i)`.
The at-risk set at time :math:`T_i` is :math:`R(T_i) = \{i' : T_{i'} \geq T_i\}`, and :math:`h_{\boldsymbol{\theta}}` is the prognostic index output by the model.


Penalty approximation
------------------------

In the sparse layer, define :math:`\gamma_j` as the main effect and :math:`\beta_{1j}, \ldots, \beta_{qj}` as the interaction effects.
The coefficient vector associated with the corresponding G and G-E variables in :math:`H` is denoted by :math:`\boldsymbol{b}_j = (\gamma_j, \beta_{1j}, \ldots, \beta_{qj})^{\top} = (b_{j1}, \ldots, b_{j(q+1)})^{\top}`, where :math:`j = 1, \ldots, p`.

Let :math:`K` denote the total number of fully connected layers in the network, and :math:`\omega_k` be the weight matrix for the :math:`k`-th layer. The objective function is:

.. math::
    L(\boldsymbol{\theta}) = l_{\cdot}(\boldsymbol{\theta}) + \sum_{j=1}^p\rho(||\boldsymbol{b_j}||;\lambda_1,s) + \sum_{j=1}^p \sum_{k=1}^q \rho(|\beta_{kj}|; \lambda_2, s) + \lambda \biggl( \sum_{k=1}^K||\omega_k||_F^2 \biggr)

where :math:`l_{\cdot}(\boldsymbol{\theta})` is the corresponding loss function depending on outcomes, :math:`|\cdot|` denotes the :math:`L_2`-norm, :math:`\rho(t; \lambda, s) = \lambda \int_0^{|t|} \left(1 - \frac{x}{\lambda s}\right)+ dx`
is the minimax concave penalty (MCP), and :math:`\lambda_1, \lambda_2 > 0` are tuning parameters. :math:`|\cdot|_{F}` denotes the Frobenius norm of a matrix, and :math:`\lambda > 0` is another tuning parameter.

We set :math:`\lambda_1 = \sqrt{q + 1} \lambda_2` and :math:`s = 3`, and use the local quadratic approximation (LQA) techniqueto optimize MCP penalties in :math:`l(\theta)`.
The two MCP penalties can be approximated as:

.. math::
    \sum_{j=1}^p\rho(\|\boldsymbol{b}_j\|;\lambda_1,s) = \sum_{j=1}^{p}\frac{\rho_{\lambda_{1}}^{\prime}\left(\|\boldsymbol{b}_j^{(m)}\|\right)}{2\|\boldsymbol{b}_j^{(m)}\|}\frac{|\gamma_{j}^{(m)}|}{\|\boldsymbol{b}_j^{(m)}\|}\gamma_{j}^{2}

.. math::
    \sum_{j=1}^p\sum_{k=1}^q\rho(|\beta_{kj}|;\lambda_2,s)=\sum_{j=1}^p\sum_{k=1}^q\left[\frac{\rho_{\lambda_1}^{\prime}\left(\|\boldsymbol{b}_j^{(m)}\|\right)}{2\|\boldsymbol{b}_j^{(m)}\|}\frac{|\beta_{kj}^{(m)}|}{\|\boldsymbol{b}_j^{(m)}\|}+\frac{\rho_{\lambda_2}^{\prime}\left(|\beta_{kj}^{(m)}|\right)}{2|\beta_{kj}^{(m)}|}\right]\beta_{kj}^2

where :math:`m` denotes the :math:`m`-th iteration, and :math:`\rho_\lambda^{\prime}(t) = (\lambda - \frac{t}{s})_+` is the first derivative of the MCP penalty.

We minimize the objective function, iterating the estimation process until the training converges and parameters stabilize folloing the algorithm below.


Algorithm: training of ScalarGE
-------------------------------

**Input**:

- Gene variables :math:`\boldsymbol{X}` and environmental variables :math:`\boldsymbol{Z}`;

- Survival output :math:`(T,\delta)` or continuous outputs :math:`y` or binary output :math:`y`;

- Learning rates of the sparse layer and the fully connected layers :math:`\{\alpha_1,\alpha_2\}`;

- Tuning and regularization parameters of the MCP penalty :math:`\{\lambda_1, \lambda_2, s\}`;

- Tuning parameter of the fully connected layers :math:`\lambda`.

**Initialize**:

- Sparse layer :math:`\boldsymbol{b}^{(0)}`, :math:`k`-th fully connected layer :math:`\omega_k^{(0)}`, :math:`m = 0`.

**Repeat**:

- Update the approximated MCP penalties with the current estimate :math:`\boldsymbol{b}^{(m)}`;

- Update Loss :math:`= l.(\boldsymbol{\theta}) + \text{approximated MCP penalties} + \lambda \sum_{k=1}^{K} \|\omega_{k}^{(m)}\|_F^2`;

- Conduct back propagation, and obtain the gradients :math:`\frac{\partial \text{Loss}}{\partial \boldsymbol{b}_j^{(m)}}` and :math:`\frac{\partial \text{Loss}}{\partial \omega_k^{(m)}}`;

- For :math:`j = 1` to :math:`p` do

  - Update estimates :math:`\gamma_j^{(m+1)} = \gamma_j^{(m)} - \alpha_1 \frac{\partial \text{Loss}}{\partial \gamma_j^{(m)}}`;

  - For :math:`k = 1` to :math:`q` do

    - Update estimates :math:`\beta_{kj}^{(m+1)} = \beta_{kj}^{(m)} - \alpha_1 \frac{\partial \text{Loss}}{\partial \beta_{kj}^{(m)}}`;

  - End for;

- End for;

- For :math:`k = 1` to :math:`K` do

  - Update :math:`\omega_k^{(m+1)} = \omega_k^{(m)} - \alpha_2 \frac{\partial \text{Loss}}{\partial \omega_k^{(m)}}`;

- End for;

- Update :math:`m = m + 1`;

Until convergence or :math:`m` reaches its maximum.
