<div align="center">
<img src="image/logo.jpg" alt="logo" width="500"></img>
</div>

[![pypi](https://img.shields.io/pypi/v/GENetLib?logo=Pypi)](https://pypi.org/project/GENetLib)
![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-lightblue.svg)
[![Build](https://github.com/Barry57/GENetLib/actions/workflows/CI.yml/badge.svg)](https://github.com/Barry57/GENetLib/actions/workflows/CI.yml/badge.svg)
[![codecov](https://codecov.io/github/Barry57/GENetLib/graph/badge.svg?token=9J9QMN7L9Z)](https://codecov.io/github/Barry57/GENetLib)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkgreen.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## `GENetLib`: A Python Library for Gene–environment Interaction Analysis via Deep Learning
``GENetLib`` is a Python library designed for gene-environment interaction analysis via neural network, addressing the analytical challenges in complex disease research. 
This package is capable of handling a variety of input data types:
- Scalar input data
- Functional input data (or densely measured data)

This package also supports diverse output requirements:
- Continuous output data
- Binary output data
- Survival output data

By integrating minimax concave penalty (MCP) and $L_2$-norm regularization within a neural network estimation framework, ``GENetLib`` offers an innovative solution for high-dimensional genetic data analysis. The framework is shown below.

<div align="center">
<img src="image/framework.png" alt="framework" width="600"></img>
</div>

We provide a web-based documentation which introduces the meaning of function parameters, the usage of functions, detailed information about methods, and gives examples for each. The web page is available at
[documentations](https://open-box.readthedocs.io/en/latest/).
This package has been uploaded to PyPI with previous versions, and the web page is available at
[PyPI package](https://pypi.org/project/genetlib/). Users can also check [tags](https://github.com/Barry57/GENetLib/releases) to get historical versions.

## Installation
It is recommended to use ``pip`` for installation.
```c
pip install GENetLib
```
For Linux or Mac users, another option is
```c
conda install GENetLib
```
To get further information about installation and independencies, please move to [installation instructions](https://open-box.readthedocs.io/en/latest/).

## Quick start
We start with the two basic functions ``scalar_ge`` and ``func_ge``.
### scalar_ge
``scalar_ge`` performs G-E interaction analysis via deep leanring when the input is scalar data.
```Python
from GENetLib.sim_data_scalar import sim_data_scalar
from GENetLib.scalar_ge import scalar_ge

# Get example data where input is scalar data and output is survival data
scalar_survival_linear = sim_data_scalar(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500,
                                         dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)

# Set up the model
scalar_ge_res = scalar_ge(scalar_survival_linear['data'], ytype = 'Survival', dim_G = 500, dim_E = 5,
                          haveGE = True, num_hidden_layers = 2, nodes_hidden_layer = [1000, 100],
                          Learning_Rate2 = 0.035, L2 = 0.1, Learning_Rate1 = 0.06, L = 0.09, Num_Epochs = 100,
                          t = 0.01, split_type = 0, ratio = [7, 3], important_feature = True, plot = True)
```
### func_ge



