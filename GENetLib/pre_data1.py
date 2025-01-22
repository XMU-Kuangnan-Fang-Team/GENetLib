import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


'''Data processing for input data when it is a whole matrix'''

def pre_data1(data, dim_G, dim_E, dim_GE = 0, ytype = 'Survival', split_type = 0, ratio = [7, 3]):

    if (split_type == 0 and len(ratio) !=2) or (split_type == 1 and len(ratio) !=3):
        raise ValueError("Split_type and ratio don't match")
    n = data.shape[0]
    
    def sort_data(data):
        if ytype == 'Survival':
            if dim_GE == 0:
                data.sort_values(by = data.columns[-2], ascending = False, inplace = True)
                x = data.loc[:, 0:dim_G-1].values
                ytime = data.iloc[:, -2].values.reshape(-1,1)
                yevent = data.iloc[:, -1].values.reshape(-1,1)
                clinical = data.loc[:, dim_G:dim_G+dim_E-1].values
                interaction = np.zeros(shape=(data.shape[0], dim_G * dim_E))
                k = 0
                for i in range(dim_E):
                    for j in range(dim_G):
                     interaction[:,k] = clinical[:,i] * x[:,j]
                     k = k + 1
            elif dim_GE > 0:
                data.sort_values(by = data.columns[-2], ascending = False, inplace = True)
                x = data.loc[:, 0:dim_G-1].values
                ytime = data.iloc[:, -2].values.reshape(-1,1)
                yevent = data.iloc[:, -1].values.reshape(-1,1)
                clinical = data.loc[:, dim_G+dim_GE:dim_G+dim_GE+dim_E-1].values
                interaction = data.loc[:, dim_G:dim_G+dim_GE-1].values
            else:
                raise ValueError("Please enter correct number of GE")
            return(x, ytime, yevent, clinical, interaction)
        elif ytype in ['Binary', 'Continuous']:
            if dim_GE == 0:
                data.sort_values(by = data.columns[-1], ascending = False, inplace = True)
                x = data.loc[:, 0:dim_G-1].values
                y = data.iloc[:, -1].values.reshape(-1,1)
                clinical = data.loc[:, dim_G:dim_G+dim_E-1].values
                interaction = np.zeros(shape=(data.shape[0], dim_G * dim_E))
                k = 0
                for i in range(dim_E):
                    for j in range(dim_G):
                     interaction[:,k] = clinical[:,i] * x[:,j]
                     k = k + 1
            elif dim_GE > 0:
                data.sort_values(by = data.columns[-1], ascending = False, inplace = True)
                x = data.loc[:, 0:dim_G-1].values
                y = data.iloc[:, -1].values.reshape(-1,1)
                clinical = data.loc[:, dim_G+dim_GE:dim_G+dim_GE+dim_E-1].values
                interaction = data.loc[:, dim_G:dim_G+dim_GE-1].values
            else:
                raise ValueError("Please enter correct number of GE")
            return(x, y, clinical, interaction)
        else:
            raise ValueError("Invalid ytype")
    
    def load_data(data, dtype):
        if ytype == 'Survival':
            x, ytime, yevent, clinical, interaction = sort_data(data)
            X = torch.from_numpy(x).type(dtype)
            YTIME = torch.from_numpy(ytime).type(dtype)
            YEVENT = torch.from_numpy(yevent).type(dtype)
            CLINICAL = torch.from_numpy(clinical).type(dtype)
            INTERACTION = torch.from_numpy(interaction).type(dtype)
            Y = [YTIME, YEVENT]
            return(X, Y, CLINICAL, INTERACTION)
        elif ytype in ['Binary', 'Continuous']:
            x, y, clinical, interaction = sort_data(data)
            X = torch.from_numpy(x).type(dtype)
            Y = torch.from_numpy(y).type(dtype)
            CLINICAL = torch.from_numpy(clinical).type(dtype)
            INTERACTION = torch.from_numpy(interaction).type(dtype)
            return(X, Y, CLINICAL, INTERACTION)
    
    if split_type == 0:
        n_train = int(n * ratio[0] / (ratio[0] + ratio[1]))
        if ytype in ['Survival','Continuous']:
            train = data.iloc[0:n_train]
            valid = data.iloc[n_train:n]
        else:
            df_0 = data[data.iloc[:, -1] == 0]
            df_1 = data[data.iloc[:, -1] == 1]
            train_0, test_0 = train_test_split(df_0, test_size = ratio[0]/(ratio[0] + ratio[1]), random_state = 42)
            train_1, test_1 = train_test_split(df_1, test_size = ratio[1]/(ratio[0] + ratio[1]), random_state = 42)
            train = pd.concat([train_0, train_1])
            valid = pd.concat([test_0, test_1])

    elif split_type == 1:
        n_train = int(n * ratio[0] / (ratio[0] + ratio[1] + ratio[2]))
        n_valid = int(n * ratio[1] / (ratio[0] + ratio[1] + ratio[2]))
        if ytype in ['Survival','Continuous']:
            train = data.iloc[0:n_train]
            valid = data.iloc[n_train:n_train+n_valid]
            test = data.iloc[n_train+n_valid:n]
        else:
            df_0 = data[data.iloc[:, -1] == 0]
            df_1 = data[data.iloc[:, -1] == 1]
            train_0, temp_0 = train_test_split(df_0, test_size = (ratio[1] + ratio[2])/(ratio[0] + ratio[1] + ratio[2]), random_state = 42)
            val_0, test_0 = train_test_split(temp_0, test_size = ratio[2]/(ratio[1] + ratio[2]), random_state = 42)
            train_1, temp_1 = train_test_split(df_1, test_size = (ratio[1] + ratio[2])/(ratio[0] + ratio[1] + ratio[2]), random_state = 42)
            val_1, test_1 = train_test_split(temp_1, test_size = ratio[2]/(ratio[1] + ratio[2]), random_state = 42)
            train = pd.concat([train_0, train_1])
            valid = pd.concat([val_0, val_1])
            test = pd.concat([test_0, test_1])
    else:
        raise ValueError("Invalid split_type")
    
    dtype = torch.FloatTensor
    if split_type == 0:
        x_train, y_train, clinical_train, interaction_train = load_data(train, dtype)
        x_valid, y_valid, clinical_valid, interaction_valid = load_data(valid, dtype)
        return (x_train, y_train, clinical_train, interaction_train,
                x_valid, y_valid, clinical_valid, interaction_valid)
    elif split_type == 1:
        x_train, y_train, clinical_train, interaction_train = load_data(train, dtype)
        x_valid, y_valid, clinical_valid, interaction_valid = load_data(valid, dtype)
        x_test, y_test, clinical_test, interaction_test = load_data(test, dtype)
        return (x_train, y_train, clinical_train, interaction_train,
                x_valid, y_valid, clinical_valid, interaction_valid,
                x_test, y_test, clinical_test, interaction_test)
    else:
        raise ValueError("Invalid split_type")

