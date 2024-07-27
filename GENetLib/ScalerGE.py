import torch
from ScalerMCP_L2Train import ScalerMCP_L2train
from PreData1 import PreData1
from PreData2 import PreData2
import pandas as pd


'''
data: list(Y, G, E, GE(optional)) or matrix(G, GE(optional), E, Y)
'''


pd.set_option('mode.chained_assignment', None)
def ScalerGE(data, ytype, dim_G, dim_E, haveGE, num_hidden_layers, nodes_hidden_layer,
             Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t = None, model = None, 
             split_type = 0, ratio = [7, 3], important_feature = True, plot = True, 
             model_reg = None, issnp = False):
    In_Nodes = dim_G
    Clinical_Nodes = dim_E
    Interaction_Nodes = dim_G * dim_E
    if haveGE == True:
        dim_GE = dim_G * dim_E
        if type(data) == list:
            y = data[0]
            x = data[1]
            clinical = data[2]
            interaction = data[3]
    else:
        dim_GE = 0
        if type(data) == list:
            y = data[0]
            x = data[1]
            clinical = data[2]
            interaction = None
    if type(data) == list:
        if split_type == 1:
            x_train, y_train, clinical_train, interaction_train,\
            x_valid, y_valid, clinical_valid, interaction_valid,\
            x_test, y_test, clinical_test, interaction_test = PreData2(y, x, clinical, interaction, ytype, split_type, ratio)
        elif split_type == 0:
            x_train, y_train, clinical_train, interaction_train,\
            x_valid, y_valid, clinical_valid, interaction_valid = PreData2(y, x, clinical, interaction, ytype, split_type, ratio)
        
    else:
        if split_type == 1:
            x_train, y_train, clinical_train, interaction_train,\
            x_valid, y_valid, clinical_valid, interaction_valid,\
            x_test, y_test, clinical_test, interaction_test = PreData1(data, dim_G, dim_E, dim_GE, ytype, split_type, ratio)
        elif split_type == 0:
            x_train, y_train, clinical_train, interaction_train,\
            x_valid, y_valid, clinical_valid, interaction_valid = PreData1(data, dim_G, dim_E, dim_GE, ytype, split_type, ratio)
    
    def important_features(tensor_, t):
        maxNum = max(abs(tensor_))
        resultPos = torch.where(abs(tensor_) > maxNum * t)[0].tolist()
        return resultPos
    
    ScalerMCP_L2trainRes = ScalerMCP_L2train(x_train, clinical_train, interaction_train, y_train,
                                             x_valid, clinical_valid, interaction_valid, y_valid,
                                             In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                             num_hidden_layers, nodes_hidden_layer, ytype, issnp,
                                             Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, plot, model, model_reg)
    if t != None:
        ifs_G = important_features(ScalerMCP_L2trainRes[4].sparse1.weight.data, t)
        ifs_GE = important_features(ScalerMCP_L2trainRes[4].sparse2.weight.data, t)
    if ytype == 'Binary':
        print('Accuracy of train:', ScalerMCP_L2trainRes[0], 
              'Accuracy of test:', ScalerMCP_L2trainRes[1]) 
        print('R2 of train:', ScalerMCP_L2trainRes[2], 
              'R2 of test:', ScalerMCP_L2trainRes[3])
        if t != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Continuous':
        print('MSE of train:', ScalerMCP_L2trainRes[0].detach().numpy()[0], 
              'MSE of test:', ScalerMCP_L2trainRes[1].detach().numpy()[0]) 
        print('R2 of train:', ScalerMCP_L2trainRes[2], 
              'R2 of test:', ScalerMCP_L2trainRes[3])
        if t != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Survival':
        print('Loss of train:', ScalerMCP_L2trainRes[0].detach().numpy()[0], 
              'Loss of test:', ScalerMCP_L2trainRes[1].detach().numpy()[0]) 
        print('C_index of train:', ScalerMCP_L2trainRes[2].detach().numpy(), 
              'C_index of test:', ScalerMCP_L2trainRes[3].detach().numpy())
        if t != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    if t != None:
        return(ScalerMCP_L2trainRes, ifs_G, ifs_GE)
    else:
        return(ScalerMCP_L2trainRes)

    
'''test'''
'''
from SimDataScaler import SimDataScaler
ytype = 'Survival'
num_hidden_layers = 2
nodes_hidden_layer = [1000, 100]
Learning_Rate2 = 0.035
L2 = 0.01
Learning_Rate1 = 0.06
L = 0.09
Num_Epochs = 100
t = 0.01
scaler_survival_linear = SimDataScaler(0.25, 0.3, 500, 5, 1500, 2, ytype, 30)
y = [scaler_survival_linear[0].iloc[:,-2], scaler_survival_linear[0].iloc[:,-1]]
x = scaler_survival_linear[0].iloc[:,0:500]
clinical = scaler_survival_linear[0].iloc[:,3000:3005]
ScalerGERes1 = ScalerGE(scaler_survival_linear[0], ytype, 500, 5, True, num_hidden_layers, nodes_hidden_layer,
                        Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, split_type = 0, ratio = [7, 3], 
                        important_feature = True, plot = True)
'''
'''
from SimDataScaler import SimDataScaler
ytype = 'Continuous'
num_hidden_layers = 2
nodes_hidden_layer = [100,10]
Learning_Rate2 = 0.035
L2 = 0.2
Learning_Rate1 = 0.06
L = 0.09
Num_Epochs = 50
t = 0.01
scaler_continuous = SimDataScaler(0.25, 0.3, 20, 5, 1500, 2, ytype, 5)
y = scaler_continuous[0].iloc[:,-1]
x = scaler_continuous[0].iloc[:,0:20]
clinical = scaler_continuous[0].iloc[:,120:125]
ScalerGERes2 = ScalerGE(scaler_continuous[0], ytype, 20, 5, True, num_hidden_layers, nodes_hidden_layer,
                        Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, split_type = 1, ratio = [3, 1, 1], 
                        important_feature = True, plot = True)
'''
'''
from SimDataScaler import SimDataScaler
ytype = 'Binary'
num_hidden_layers = 2
nodes_hidden_layer = [100,10]
Learning_Rate2 = 0.035
L2 = 0.01
Learning_Rate1 = 0.07
L = 0.07
Num_Epochs = 50
t = 0.01
dim_G = 50
n_inter = 3
scaler_binary = SimDataScaler(0.25, 0.3, dim_G, 5, 50, 2, ytype, n_inter)
y = scaler_binary[0].iloc[:,-1]
x = scaler_binary[0].iloc[:,0:20]
clinical = scaler_binary[0].iloc[:,120:125]
ScalerGERes3 = ScalerGE(scaler_binary[0], ytype, dim_G, 5, True, num_hidden_layers, nodes_hidden_layer,
                        Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, t, split_type = 1, ratio = [3, 1, 1], 
                        important_feature = True, plot = True)
'''
