# GENetLib: A Python Library for Gene–environment Interaction Analysis via Deep Learning
## Introduction
GENetLib is a Python library that addresses the lack of portable and friendly software for analyzing gene-environment (G-E) interactions using a deep learning approach with penalization, as developed by Wu et al., 2023. It also provides a functional data analysis moethod based on neural network framework.

***References*** <br>
Wu, S., Xu, Y., Zhang, Q., & Ma, S. (2023). Gene–environment interaction analysis via deep learning. Genetic Epidemiology, 1–26. https://doi.org/10.1002/gepi.22518 <br>
Ren, R., Fang, K., Zhang, Q., & Ma, S. (2023). FunctanSNP: an R package for functional analysis of dense SNP data (with interactions). Bioinformatics, 39(12), btad741. https://doi.org/10.1093/bioinformatics/btad741

## Installation
### Requirements
matplotlib==3.7.1<br />
numpy==1.24.3<br />
pandas==1.5.3<br />
scipy==1.10.1<br />
setuptools==67.8.0<br />
torch==2.3.0<br />
### Normal installation
```c
pip install GENetLib
```
### Mirror
```c
pip install GENetLib -i https://pypi.tuna.tsinghua.edu.cn/simple 
```
## Functions
### Menu
- [sim_data_scaler](#sim_data_scaler)
- [sim_data_func](#sim_data_func)
- [scaler_ge](#scaler_ge)
- [func_ge](#func_ge)
- [grid_scaler_ge](#grid_scaler_ge)
- [grid_func_ge](#grid_func_ge)

#### sim_data_scaler
*Example data for method scaler_ge and grid_scaler_ge*
##### Description
Example data for users to apply the method scaler_ge and grid_scaler_ge.
##### Usage
```c
sim_data_scaler(rho_G, rho_E, dim_G, dim_E, n, dim_E_Sparse = 0, ytype = 'Survival', n_inter = None, linear = True, seed = 0)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
rho_G|numeric, correlation of gene variables.
rho_E|numeric, correlation of environment variables.
dim_G|numeric, dimension of gene variables.
dim_E|numeric, dimension of environment variables.
n|numeric, sample size.
dim_E_Sparse|numeric, dimension of sparse environment variables.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y. If not specified, the default is survival.
n_inter|numeric, number of interaction effect variables.
linear|bool, "True" or "False", whether or not to generate linear data. The default is True.
seed|numeric, random seeds each time when data is generated.
##### Value
The function "sim_data_scaler" outputs a tuple including generated data and the positions of interaction effect variables.
- data: a dataframe contains gene variables, environment variables, interaction variables and output y.
- interaction efecct variables: an array contains the positions of interaction effect variables.
##### See Also
See also as [scaler_ge](#scaler_ge), [grid_scaler_ge](#grid_scaler_ge).
##### Examples
```c
import GENetLib
from GENetLib.sim_data_scaler import sim_data_scaler
scaler_survival_linear = sim_data_scaler(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500, dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)
scaler_survival_linear_data = scaler_survival_linear[0]
scaler_survival_linear_inter = scaler_survival_linear[1]
```
<br />
<br />

#### sim_data_func
*Example data for method func_ge and grid_func_ge*
##### Description
Example data for users to apply the method func_ge and grid_func_ge.
##### Usage
```c
sim_data_func(n, m, ytype, seed = 0)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
n|numeric, sample size.
m|numeric, the sequence length of each sample.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y. If not specified, the default is continuous.
seed|numeric, random seeds each time when data is generated.
##### Value
The function "sim_data_scaler" outputs a dictionary including response variable y, scalar variable z and sequence (genotypes) data X.
- x: a matrix representing the sequence data, with the number of rows equal to the number of samples.
- y: an array representing the response variables.
- z: a matrix representing the scalar covariates, with the number of rows equal to the number of samples.
- location: a list defining the sampling sites of the sequence (genotypes) data.
##### See Also
See also as [func_ge](#func_ge), [grid_func_ge](#grid_func_ge).
##### Examples
```c
import GENetLib
from GENetLib.sim_data_func import sim_data_func
func_continuous = sim_data_func(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
x = func_continuous['X']
y = func_continuous['y']
z = func_continuous['z']
location = func_continuous['location']
```
<br />
<br />

#### scaler_ge
*G-E interaction analysis via deep leanring when the input X is scaler*
##### Description
This function provides an approach based on deep neural network in conjunction with MCP and L2 penalizations which can simultaneously conduct model estimation and selection of important main G effects and G–E interactions, while uniquely respecting the "main effects, interactions" variable selection hierarchy.
##### Usage
```c
scaler_ge(data, ytype, dim_G, dim_E, haveGE, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t = None, model = None, split_type = 0, ratio = [7, 3], important_feature = True, plot = True, model_reg = None, isfunc = False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
data|dataframe or list, follow the format: dataframe with {G, GE(optional), E, y} or list with {y, G, E, GE(optional)}.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y.
dim_G|numeric, dimension of gene variables.
dim_E|numeric, dimension of environment variables.
haveGE|bool, "True" or "False", whether there are GE interactions in the data. If not, the function will calculate GE interactions.
num_hidden_layers|numeric, number of hidden layers in the neural network.
nodes_hidden_layer|list, contains number of nodes in each hidden layer.
Learning_Rate2|numeric, learning rate of hidden layers.
L2|numeric, tuning parameter of L2 penalization.
Learning_Rate1|numeric, learning rate of sparse layers.
L|numeric, tuning parameter of MCP penalization.
Num_Epochs|numeric, number of epochs for neural network training.
t|numeric, threshold in the selection ofimportant features.
model|tuple, pre-trained models. If not specified, the default is none.
split_type|integer, types of data split. If split_type = 0, the data is divided into a training set and a validation set. If split_type = 1, the data is divided into a training set, a validation set and a test set.
ratio|list, the ratio of data split.
important_feature|bool, "True" or "False", whether or not to show output features.
plot|bool, "True" or "False", whether or not to show the line plot of residuals with the number of neural network epochs.
##### Value
The function "scaler_ge" outputs a tuple including training results of the neural network.
- Residual of the training set.
- Residual of the validation set.
- C index(y is survival) or R2(y is continuous or binary) of the training set.
- C index(y is survival) or R2(y is continuous or binary) of the validation set.
- A neural network after training.
- Important features of gene variables.
- Important features of G-E interaction variables.
##### See Also
See also as [sim_data_scaler](#sim_data_scaler), [grid_scaler_ge](#grid_scaler_ge).
##### Examples
```c
import GENetLib
from GENetLib.sim_data_scaler import sim_data_scaler
from GENetLib.scaler_ge import scaler_ge
ytype = 'Survival'
num_hidden_layers = 2
nodes_hidden_layer = [1000, 100]
Learning_Rate2 = 0.035
L2 = 0.1
Learning_Rate1 = 0.06
L = 0.09
Num_Epochs = 100
t = 0.01
scaler_survival_linear = sim_data_scaler(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500, dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)
scaler_ge_res = scaler_ge(scaler_survival_linear[0], ytype, 500, 5, True, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, split_type = 0, ratio = [7, 3], important_feature = True, plot = True)
```
<br />
<br />

#### func_ge
*G-E interaction analysis via deep leanring when the input X is functional data*
##### Description
This function provides an approach based on deep neural network in conjunction with MCP and L2 penalizations which treats functional data or discrete realizations of functioanal data.
##### Usage
```c
func_ge(y, z, location, X, ytype, btype, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, t = None, Bsplines = 20, norder1 = 4, model = None, split_type = 0, ratio = [7, 3], plot_res = True, plot_beta = True)
```
|Arguments|Description|
|:---:|:---:|
y|numeric, an array representing the response variables.
z|numeric, a matrix representing the scalar covariates, with the number of rows equal to the number of samples.
location|list, a list defining the sampling sites of the sequence data.
X|numeric or dict, a matrix representing the sequence data with the number of rows equal to the number of samples or a "fd" item which represents the functional data.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y.
btype|character, "Bspline", "Exponential", "Fourier", "Monomial" or "power" type of spline.
num_hidden_layers|numeric, number of hidden layers in the neural network.
nodes_hidden_layer|list, contains number of nodes in each hidden layer.
Learning_Rate2|numeric, learning rate of hidden layers.
L2|numeric, tuning parameter of L2 penalization.
Learning_Rate1|numeric, learning rate of sparse layers.
L|numeric, tuning parameter of MCP penalization.
Num_Epochs|numeric, number of epochs for neural network training.
nbasis1|integer, an integer specifying the number of basis functions that constitutes the genetic variation function.
params1|integer, in addition to rangeval1 (a vector of length 2 giving the lower and upper limits of the range of permissible values for the genetic variation function) and nbasis1, all bases have one or two parameters unique to that basis type or shared with one other.
Bsplines|integer, an integer specifying the number of basis functions that constitutes the genetic effect function.
norder1|integer, an integer specifying the order of bsplines that constitutes the genetic effect function, which is one higher than their degree. The default of 4 gives cubic splines.
model|tuple, pre-trained models. If not specified, the default is none.
split_type|integer, types of data split. If split_type = 0, the data is divided into a training set and a validation set. If split_type = 1, the data is divided into a training set, a validation set and a test set.
ratio|list, the ratio of data split.
plot_res|bool, "True" or "False", whether or not to show the line plot of residuals with the number of neural network epochs.
plot_beta|bool, "True" or "False", whether or not to show the graph of predicted functions.
##### Value
The function "func_ge" outputs a tuple including training results of the neural network.
- Residual of the training set.
- Residual of the validation set.
- C index(y is survival) or R2(y is continuous or binary) of the training set.
- C index(y is survival) or R2(y is continuous or binary) of the validation set.
- A neural network after training.
- Estimated coefficients of the chosen basis functions for the genetic effect function beta0(t) and interaction items betak(t).
- The estimated genetic effect function beta(t) and interaction items betak(t).
##### See Also
See also as [sim_data_func](#sim_data_func), [grid_func_ge](#grid_func_ge).
##### Examples
```c
import GENetLib
from GENetLib.sim_data_func import sim_data_func
from GENetLib.func_ge import func_ge
num_hidden_layers = 2
nodes_hidden_layer = [100,10]
Learning_Rate2 = 0.035
L2 = 0.01
Learning_Rate1 = 0.02
L = 0.01
Num_Epochs = 50
nbasis1 = 5
params1 = 4
func_continuous = sim_data_func(n = 1500, m = 30, ytype = 'Continuous', seed = 123)
y = func_continuous['y']
z = func_continuous['z']
location = func_continuous['location']
X = func_continuous['X']
func_ge_res = func_ge(y, z, location, X, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, Bsplines = 5, norder1 = 4, model = None, split_type = 1, ratio = [3, 1, 1], plot_res = True)
```
<br />
<br />

#### grid_scaler_ge
*Grid search for scaler_ge*
##### Description
This function performs grid search for scaler_ge over a grid of values for the regularization parameter L, L2 and learning rate Learning_Rate1, Learning_Rate2.
##### Usage
```c
grid_scaler_ge(data, ytype, dim_G, dim_E, haveGE, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t = None, model = None, split_type = 0, ratio = [7, 3], important_feature = True, plot = True, model_reg = None, isfunc = False)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
data|dataframe or list, follow the format: dataframe with {G, GE(optional), E, y} or list with {y, G, E, GE(optional)}.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y.
dim_G|numeric, dimension of gene variables.
dim_E|numeric, dimension of environment variables.
haveGE|bool, "True" or "False", whether there are GE interactions in the data. If not, the function will calculate GE interactions.
num_hidden_layers|numeric, number of hidden layers in the neural network.
nodes_hidden_layer|list, contains number of nodes in each hidden layer.
Learning_Rate2|list, learning rate of hidden layers.
L2|list, tuning parameter of L2 penalization.
Learning_Rate1|list, learning rate of sparse layers.
L|list, tuning parameter of MCP penalization.
Num_Epochs|numeric, number of epochs for neural network training.
t|numeric, threshold in the selection ofimportant features.
model|tuple, pre-trained models. If not specified, the default is none.
split_type|integer, types of data split. If split_type = 0, the data is divided into a training set and a validation set. If split_type = 1, the data is divided into a training set, a validation set and a test set.
ratio|list, the ratio of data split.
important_feature|bool, "True" or "False", whether or not to show output features.
plot|bool, "True" or "False", whether or not to show the line plot of residuals with the number of neural network epochs.
##### Value
The function "grid_scaler_ge" outputs a tuple including training results of the neural network.
- Values of tunning parameters after grid search.
- Residual of the training set.
- Residual of the validation set.
- C index(y is survival) or R2(y is continuous or binary) of the training set.
- C index(y is survival) or R2(y is continuous or binary) of the validation set.
- A neural network after training.
- Important features of gene variables.
- Important features of GE interaction variables.
##### See Also
See also as [sim_data_scaler](#sim_data_scaler), [scaler_ge](#scaler_ge).
##### Examples
```c
import GENetLib
from GENetLib.sim_data_scaler import sim_data_scaler
from GENetLib.grid_scaler_ge import grid_scaler_ge
ytype = 'Survival'
num_hidden_layers = 2
nodes_hidden_layer = [1000, 100]
Learning_Rate2 = [0.035, 0.045]
L2 = [0.1]
Learning_Rate1 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
L = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
Num_Epochs = 100
t = 0.01
dim_E = 5
dim_G = 500
haveGE = True
scaler_survival_linear = sim_data_scaler(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500, dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)
grid_scaler_ge_res = grid_scaler_ge(scaler_survival_linear[0], ytype, dim_G, dim_E, haveGE, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, split_type = 1, ratio = [3, 1, 1], plot = True)
```
<br />
<br />

#### grid_func_ge
*Grid search for func_ge*
##### Description
This function performs grid search for func_ge over a grid of values for the regularization parameter L, L2 and learning rate Learning_Rate1, Learning_Rate2.
##### Usage
```c
grid_func_ge(y, z, location, X, ytype, btype, num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, t = None, Bsplines = 20, norder1 = 4, model = None, split_type = 0, ratio = [7, 3], plot_res = True, plot_beta = True)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
y|numeric, an array representing the response variables.
z|numeric, a matrix representing the scalar covariates, with the number of rows equal to the number of samples.
location|list, a list defining the sampling sites of the sequence data.
X|numeric or dict, a matrix representing the sequence data with the number of rows equal to the number of samples or a "fd" item which represents the functional data.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y.
btype|character, "Bspline", "Exponential", "Fourier", "Monomial" or "power" type of spline.
num_hidden_layers|numeric, number of hidden layers in the neural network.
nodes_hidden_layer|list, contains number of nodes in each hidden layer.
Learning_Rate2|list, learning rate of hidden layers.
L2|list, tuning parameter of L2 penalization.
Learning_Rate1|list, learning rate of sparse layers.
L|list, tuning parameter of MCP penalization.
Num_Epochs|numeric, number of epochs for neural network training.
nbasis1|integer, an integer specifying the number of basis functions that constitutes the genetic variation function.
params1|integer, in addition to rangeval1 (a vector of length 2 giving the lower and upper limits of the range of permissible values for the genetic variation function) and nbasis1, all bases have one or two parameters unique to that basis type or shared with one other.
Bsplines|integer, an integer specifying the number of basis functions that constitutes the genetic effect function.
norder1|integer, an integer specifying the order of bsplines that constitutes the genetic effect function, which is one higher than their degree. The default of 4 gives cubic splines.
model|tuple, pre-trained models. If not specified, the default is none.
split_type|integer, types of data split. If split_type = 0, the data is divided into a training set and a validation set. If split_type = 1, the data is divided into a training set, a validation set and a test set.
ratio|list, the ratio of data split.
plot_res|bool, "True" or "False", whether or not to show the line plot of residuals with the number of neural network epochs.
plot_beta|bool, "True" or "False", whether or not to show the graph of predicted functions.
##### Value
The function "grid_func_ge" outputs a tuple including training results of the neural network.
- Values of tunning parameters after grid search.
- Residual of the training set.
- Residual of the validation set.
- C index(y is survival) or R2(y is continuous or binary) of the training set.
- C index(y is survival) or R2(y is continuous or binary) of the validation set.
- A neural network after training.
- Estimated coefficients of the chosen basis functions for the genetic effect function beta0(t) and interaction items betak(t).
- The estimated genetic effect function beta(t) and interaction items betak(t).
##### See Also
See also as [sim_data_func](#sim_data_func), [func_ge](#func_ge).
##### Examples
```c
import GENetLib
from GENetLib.sim_data_func import sim_data_func
from GENetLib.grid_func_ge import grid_func_ge
num_hidden_layers = 2
nodes_hidden_layer = [100, 10]
Learning_Rate2 = [0.005, 0.01, 0.015]
L2 = [0.005, 0.01, 0.015]
Learning_Rate1 = [0.001, 0.005]
L = [0.005, 0.006, 0.007]
Num_Epochs = 50
nbasis1 = 5
params1 = 4
func_continuous = sim_data_func(n = 1000, m = 30, ytype = 'Continuous', seed = 1)
y = func_continuous['y']
z = func_continuous['z']
location = func_continuous['location']
X = func_continuous['X']
grid_func_ge_res = grid_func_ge(y, z, location, X, 'Continuous', 'Bspline', num_hidden_layers, nodes_hidden_layer, Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, nbasis1, params1, Bsplines = 5, norder1 = 4, model = None, split_type = 0, ratio = [7,3], plot_res = True)
```
<br />
<br />
