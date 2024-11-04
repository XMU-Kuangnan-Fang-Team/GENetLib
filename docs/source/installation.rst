Installation
============


Install from PyPI
------------------
It is recommended to use ``pip`` for installation. You need to check the upgrade at the beginning:

.. code-block:: console

   pip install --upgrade pip

Then you can instll this package using the code:

.. code-block:: console

   pip install GENetLib

Install from source
---------------------

This is recommended if you want to work with the latest development version or if you wish to contribute to skscope.


Then, clone the latest source code from GitHub and enter the directory of


For more details, or for installing nightly versions, see the
`TorchRL installation guide <https://github.com/pytorch/rl#installation>`__.

Install BenchMARL
-----------------

You can just install it from PyPi

.. code-block:: console

   pip install benchmarl

Or also clone it locally to access the configs and scripts

.. code-block:: console

    git clone https://github.com/facebookresearch/BenchMARL.git
    pip install -e BenchMARL

Install optional packages
-------------------------

By default, BenchMARL has only the core requirements.
Here are some optional packages you may want to install.

Logging
^^^^^^^

You may want to install the following rendering and logging tools

.. code-block:: console

   pip install wandb moviepy torchvision av

Install environments
--------------------

All environment dependencies are optional in BenchMARL and can be installed separately.

VMAS
^^^^
:github:`null` `GitHub <https://github.com/proroklab/VectorizedMultiAgentSimulator>`__

.. code-block:: console

   pip install vmas

PettingZoo
^^^^^^^^^^
:github:`null` `GitHub <https://github.com/Farama-Foundation/PettingZoo>`__


.. code-block:: console

   pip install "pettingzoo[all]"

MeltingPot
^^^^^^^^^^
:github:`null` `GitHub <https://github.com/google-deepmind/meltingpot>`__


.. code-block:: console

   pip install "dm-meltingpot"


matplotlib==3.7.1<br />
numpy==1.24.3<br />
pandas==1.5.3<br />
scipy==1.10.1<br />
setuptools==67.8.0<br />
torch==2.3.0<br />