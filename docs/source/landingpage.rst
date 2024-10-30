.. raw:: html

    <div align="center">

.. image:: source/_static/logo.jpg
   :alt: logo
   :width: 500

.. raw:: html

    </div>

.. image:: https://img.shields.io/pypi/v/GENetLib?logo=Pypi
    :target: https://pypi.org/project/GENetLib
    :alt: PyPIversion
.. image:: https://img.shields.io/badge/Python-3.8%2B-lightblue.svg
    :target: https://pypi.org/project/GENetLib
    :alt: Pythonversion
.. image:: https://github.com/Barry57/GENetLib/actions/workflows/CI.yml/badge.svg
    :target: https://github.com/Barry57/GENetLib/actions/workflows/CI.yml/badge.svg
    :alt: CI
.. image:: https://codecov.io/github/Barry57/GENetLib/graph/badge.svg?token=9J9QMN7L9Z
    :target: https://codecov.io/github/Barry57/GENetLib
    :alt: Codecov
.. image:: https://img.shields.io/badge/License-MIT-darkgreen.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License
.. image:: https://img.shields.io/badge/License-MIT-darkgreen.svg
    :target: https://opensource.org/licenses/MIT
    :alt: Codestyle


``GENetLib``: A Python Library for Gene–environment Interaction Analysis via Deep Learning
================================================================================

``GENetLib`` is a Python library designed for gene-environment interaction analysis via neural network, addressing the analytical challenges in complex disease research. This package is capable of handling a variety of input data types:

- Scalar input data
- Functional input data (or densely measured data)

This package also supports diverse output requirements:

- Continuous output data
- Binary output data
- Survival output data

By integrating minimax concave penalty (MCP) and L2-norm regularization within a neural network estimation framework, ``GENetLib`` offers an innovative solution for high-dimensional genetic data analysis. The framework is shown below.

.. image:: image/framework.png
   :alt: framework
   :width: 600

We provide a web-based documentation which introduces the meaning of function parameters, the usage of functions, detailed information about methods, and gives examples for each. The web page is available at `documentations <https://open-box.readthedocs.io/en/latest/>`_. This package has been uploaded to PyPI with previous versions, and the web page is available at `PyPI package <https://pypi.org/project/genetlib/>`_. Users can also check `tags <https://github.com/Barry57/GENetLib/releases>`_ to get historical versions.
