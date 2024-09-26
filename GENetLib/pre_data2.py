import numpy as np
import pandas as pd

from GENetLib.pre_data1 import pre_data1


'''Data processing for input data when it is divided'''

def pre_data2(y, x, clinical, interaction = None, ytype = 'Survival', split_type = 0, ratio = [7, 3]):
    
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
    return(pre_data1(data, dim_G, dim_E, dim_GE, ytype, split_type, ratio))

