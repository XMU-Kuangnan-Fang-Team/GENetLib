Methods
======================================

In this section, we introduce the main models in **GENetLib** detailedly.

First, we present the neural network method designed for flexible G-E interaction analysis across diverse data types:

.. image:: /_static/GENet_framework.png
   :width: 700
   :align: center

We made a neural network with a sparse layer, some hidden layers, and an output layer.
The sparse layer connects each input directly, including genetic (G), environmental (E), and G-E interaction variables.
We use a penalty to pick important G and G-E variables, but not E variables because they are usually not many.
The hidden layers can have different setups, and we show two here.
The first layer combines the chosen G effects, G-E interactions, and all E effects.
Later layers work on what the previous layers give.
We use ReLU to make learning better and avoid gradient vanishing.
The output layer changes based on the type of result and uses a special loss function for that.

Next we introduce the ScalarGE model and FuncGE model, which stand for scalar input and functional input separately.

.. toctree::
   :maxdepth: 1

   ScalarGE
   FuncGE