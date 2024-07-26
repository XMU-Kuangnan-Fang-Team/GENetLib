# GENetLib: A Python Library for Gene–environment Interaction Analysis via Deep Learning
## Introduction
GENetLib is a Python library that addresses the lack of portable and friendly software for analyzing gene-environment (G-E) interactions using a deep learning approach with penalization, as developed by Wu et al., 2023. It also tackles the challenge of high-dimensional SNP data analysis by employing a functional data analysis method that reduces data dimensionality, which builds upon with a G-E interaction approach proposed by Ren et al., 2023. 
***References:***
Wu, S., Xu, Y., Zhang, Q., & Ma, S. (2023). Gene–environment interaction analysis via deep learning. Genetic Epidemiology, 1–26. https://doi.org/10.1002/gepi.22518
Ren, R., Fang, K., Zhang, Q., & Ma, S. (2023). FunctanSNP: an R package for functional analysis of dense SNP data (with interactions). Bioinformatics, 39(12), btad741. https://doi.org/10.1093/bioinformatics/btad741

## Installation
### Requirements
matplotlib==3.7.1<br />
numpy==1.24.3<br />
pandas==1.5.3<br />
scikit_learn==1.2.2<br />
scipy==1.10.1<br />
setuptools==67.8.0<br />
torch==2.3.0<br />
### Normal installation
```c
pip install GENetLib
```
### Mirror
```c
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple GENetLib
```
## Functions
### Menu
- [SimDataScaler](#SimDataScaler)
- [SimDataSNP](#SimDataSNP)
- [ScalerGE](#ScalerGE)
- [SNPGE](#SNPGE)
- [GridScalerGE](#GridScalerGE)
- [GridSNPGE](#GridSNPGE)

#### SimDataScaler
*Example data for method ScalerGE and GridScalerGE*
##### Description
Example data for users to apply the method ScalerGE and GridScalerGE.
##### Usage
```c
SimDataScaler(rho_G, rho_E, dim_G, dim_E, n, dim_E_Sparse = 0, ytype = 'Survival', n_inter = None, linear = True, seed = 0)
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
The function "SimDataScaler" outputs a tuple including generated data and the positions of interaction effect variables.
- part 1: a dataframe contains gene variables, environment variables, interaction variables and output y.
- part 2: an array contains the positions of interaction effect variables.
##### See Also
See also as [ScalerGE](#ScalerGE), [GridScalerGE](#GridScalerGE).
##### Examples
```c
rho_G, rho_E, dim_G, dim_E, n, dim_E_Sparse = 0, ytype = 'Survival', n_inter = None, linear = True, seed = 0
scaler_survival_linear = SimDataScaler(rho_G = 0.25, rho_E = 0.3, dim_G = 500, dim_E = 5, n = 1500, dim_E_Sparse = 2, ytype = 'Survival', n_inter = 30)
scaler_survival_linear_data = scaler_survival_linear[0]
scaler_survival_linear_inter = scaler_survival_linear[1]
```
<br />
<br />

#### SimDataSNP
*Example data for method SNPGE and GridSNPGE*
##### Description
Example data for users to apply the method SNPGE and GridSNPGE.
##### Usage
```c
SimDataSNP(n, m, ytype, seed = 0)
```
##### Arguments
|Arguments|Description|
|:---:|:---:|
n|numeric, sample size.
m|numeric, the sequence length of each sample.
ytype|character, "Survival", "Binary" or "Continuous" type of the output y. If not specified, the default is continuous.
seed|numeric, random seeds each time when data is generated.
##### Value
The function "SimDataScaler" outputs a dictionary including response variable y, scalar variable z and sequence (genotypes) data X.
- x: a matrix representing the sequence data, with the number of rows equal to the number of samples.
- y: an array representing the response variables..
- z: a matrix representing the scalar covariates, with the number of rows equal to the number of samples.
- location: a numeric vector defining the sampling sites of the sequence (genotypes) data.
##### See Also
See also as [SNPGE](#SNPGE), [GridSNPGE](#GridSNPGE).
##### Examples
```c
snp_continuous = SimDataSNP(n = 1000, m = 100, ytype = 'Continuous', seed = 1)
x = snp_continuous['X']
y = snp_continuous['y']
z = snp_continuous['z']
location = snp_continuous['location']
```
<br />
<br />



