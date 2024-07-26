# GENetLib: A Python Library for Gene–environment Interaction Analysis via Deep Learning
## Introduction
The paper introduces GENetLib, a Python library that addresses the lack of portable and friendly software for analyzing gene-environment (G-E) interactions using a deep learning approach with penalization, as developed by Wu et al., 2023. It also tackles the challenge of high-dimensional SNP data analysis by employing a functional data analysis method that reduces data dimensionality, as demonstrated effective in studies by Fan et al., 2013, and Zhang et al., 2016, and further builds upon this with a G-E interaction approach proposed by Ren et al., 2023. This method has been rigorously tested with both simulated and actual data to ensure its precision and effectiveness, allowing users to easily construct G-E interaction models with SNP data or scalar genetic data using GENetLib.

***References:***

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


