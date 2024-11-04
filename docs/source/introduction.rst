Introduction
========================================================

.. image:: _static/logo.jpg
   :width: 700
   :align: center


.. image:: https://img.shields.io/pypi/v/GENetLib?logo=Pypi
   :target: https://pypi.org/project/GENetLib
.. image:: https://img.shields.io/badge/Python-3.8%2B-lightblue.svg
.. image:: https://github.com/Barry57/GENetLib/actions/workflows/CI.yml/badge.svg
   :target: https://github.com/Barry57/GENetLib/actions/workflows/CI.yml/badge.svg
.. image:: https://codecov.io/github/Barry57/GENetLib/graph/badge.svg?token=9J9QMN7L9Z
   :target: https://codecov.io/github/Barry57/GENetLib
.. image:: https://img.shields.io/badge/License-MIT-darkgreen.svg
   :target: https://opensource.org/licenses/MIT
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

`GENetLib` is a Python library designed for gene-environment (G-E) interaction analysis via neural network, 
addressing the analytical challenges in complex disease research. 
This package is capable of handling a variety of input data types:

- Scalar input data

- Functional input data (or densely measured data)

When the input data is scalar data, we adopt the `ScalarGE model <>`. is designed to characterize G-E interaction effects between
high-dimensional (scalar) genetic variables and environmental variables.

This package also supports diverse output requirements:

- Continuous output data

- Binary output data

- Survival output data

By integrating minimax concave penalty (MCP) and \( L_2 \)-norm regularization within a neural network estimation framework, 
`GENetLib` offers an innovative solution for high-dimensional genetic data analysis. The framework is shown below.

.. image:: _static/framework.png
   :width: 700
   :align: center

When the input data is scalar data, we adopt the `ScalarGE model <https://genetlib.readthedocs.io/en/latest/methods/ScalarGE.html>`. is designed to characterize G-E interaction effects between
high-dimensional (scalar) genetic variables and environmental variables.

This package has been uploaded to PyPI with previous versions, and the web page is available at
`PyPI package <https://pypi.org/project/genetlib/>`_.  Users can also check `tags <https://github.com/Barry57/GENetLib/releases>`_  to get historical versions.

Features
-----------

`GENetLib` has the following features:

- **Comprehensiveness**: Supports a variety of input and output formats, enabling the construction of comprehensive neural network models for G-E interaction analysis.

- **Flexibility**: Offers a multitude of parameters allowing users to build models flexibly according to their specific needs.

- **Functional data compatibility**: Implements methods for functional data analysis (FDA) in Python, facilitating the processing of functional data with Python.

- **Scalability**: New methods for G-E interaction analysis via deep learning can be easily integrated into the system.