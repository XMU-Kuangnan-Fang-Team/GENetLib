import numpy as np
import pandas as pd
from PreData1 import PreData1


'''
Data preprocessing for seperated G/E or G/GE/E
split_type = 0: train and valid; split_type = 1: train, valid and test
y is a list when ytype = 'Survival'; x and clinical are dataframe
'''


def PreData2(y, x, clinical, interaction = None, ytype = 'Survival', split_type = 0, ratio = [7, 3]):
    if (split_type == 0 and len(ratio) !=2) or (split_type == 1 and len(ratio) !=3):
        raise ValueError("Split_type and ratio don't match")
    n = x.shape[0]
    dim_G = x.shape[1]
    dim_E = clinical.shape[1]
    if ytype == 'Survival': 
        if interaction == None:
            dim_GE = 0
            data = pd.DataFrame(np.hstack((x, clinical, np.array(y).reshape(n,-1))))
        else:
            dim_GE = interaction.shape[1]
            data = pd.DataFrame(np.hstack((x, interaction, clinical, np.array(y).reshape(n,-1))))
    elif ytype in ['Binary', 'Continuous']:
        if interaction == None:
            dim_GE = 0
            data = pd.DataFrame(np.hstack((x, clinical, np.array(y).reshape(n,-1))))
        else:
            dim_GE = interaction.shape[1]
            data = pd.DataFrame(np.hstack((x, interaction, clinical, np.array(y).reshape(n,-1))))
    else:
        raise ValueError("Invalid ytype")
    return(PreData1(data, dim_G, dim_E, dim_GE, ytype, split_type, ratio))


'''
test

from SimDataScaler import SimDataScaler
scaler_survival_linear = SimDataScaler(0.25, 0.3, 500, 5, 1500, 2, 'Survival', 30)
y = [scaler_survival_linear.iloc[:,-2], scaler_survival_linear.iloc[:,-1]]
x = scaler_survival_linear.iloc[:,0:500]
clinical = scaler_survival_linear.iloc[:,3000:3005]
x_train, y_train, clinical_train, interaction_train,\
x_valid, y_valid, clinical_valid, interaction_valid,\
x_test, y_test, clinical_test, interaction_test = PreData2(y, x, clinical, split_type = 1, ratio = [3, 1, 1])
'''

