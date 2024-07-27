import pandas as pd
import torch

from GENetLib.PreData1 import PreData1
from GENetLib.PreData2 import PreData2
from GENetLib.ScalerL2Train import ScalerL2train
from GENetLib.ScalerMCP_L2Train import ScalerMCP_L2train


pd.set_option('mode.chained_assignment', None)
def GridScalerGE(data, ytype, dim_G, dim_E, haveGE, num_hidden_layers, nodes_hidden_layer,
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
    
    opt_loss = torch.Tensor([float("Inf")])
    for lr2 in Learning_Rate2:
        for l2 in L2:
            loss_train, loss_valid, index_tr, index_va, model = ScalerL2train(x_train, clinical_train, interaction_train, y_train,
                                                                              x_valid, clinical_valid, interaction_valid, y_valid,
                                                                              In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                                                              num_hidden_layers, nodes_hidden_layer, ytype, issnp,
                                                                              lr2, l2, Num_Epochs, model_reg)
            if loss_valid < opt_loss:
                opt_L2 = l2
                opt_Learning_Rate2 = lr2
                best_model = model
                opt_loss = loss_valid

    opt_loss = torch.Tensor([float("Inf")])
    GENet = None
    for l in L:
        for lrMCP in Learning_Rate1:
            loss_train, loss_valid, index_tr, index_va, MCPNet = ScalerMCP_L2train(x_train, clinical_train, interaction_train, y_train,
                                                                                   x_valid, clinical_valid, interaction_valid, y_valid,
                                                                                   In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                                                                   num_hidden_layers, nodes_hidden_layer, ytype, issnp,
                                                                                   opt_Learning_Rate2, opt_L2, lrMCP, l, Num_Epochs, plot, best_model, model_reg)
            if loss_valid < opt_loss:
                opt_loss = loss_valid
                GENet = MCPNet
                opt_l = l
                opt_lr = lrMCP
                opt_loss_train = loss_train
                opt_index_tr = index_tr
                opt_index_va = index_va

    def important_features(tensor_, t):
        maxNum = max(abs(tensor_))
        resultPos = torch.where(abs(tensor_) > maxNum * t)[0].tolist()
        return resultPos
    if t != None:
        tensor1 = GENet.sparse1.weight.data
        tensor2 = GENet.sparse2.weight.data
        ifs_G = important_features(tensor1, t)
        ifs_GE = important_features(tensor2, t)
    if ytype == 'Binary':
        print('opt_index: L2:', opt_L2, "LR2:", opt_Learning_Rate2, "LR_MCP:", opt_lr, 'L:', opt_l)
        print('Accuracy of train:', opt_loss_train, 'Accuracy of test:', opt_loss) 
        print('R2 of train:', opt_index_tr, 'R2 of test:', opt_index_va)
        if t != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Continuous':
        print('opt_index: L2:', opt_L2, "LR2:", opt_Learning_Rate2, "LR_MCP:", opt_lr, 'L:', opt_l)
        print('MSE of train:', opt_loss_train.detach().numpy()[0], 'MSE of test:', opt_loss.detach().numpy()[0]) 
        print('R2 of train:', opt_index_tr, 'R2 of test:', opt_index_va)
        if t != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Survival':
        print('opt_index: L2:', opt_L2, "LR2:", opt_Learning_Rate2, "LR_MCP:", opt_lr, 'L:', opt_l)
        print('Loss of train:', opt_loss_train.detach().numpy(), 
              'Loss of test:', opt_loss.detach().numpy()) 
        print('C_index of train:', opt_index_tr.detach().numpy(), 
              'C_index of test:', opt_index_va.detach().numpy())
        if t != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    if t != None:
        return([opt_L2, opt_Learning_Rate2, opt_lr, opt_l],opt_loss_train, opt_loss, opt_index_tr, opt_index_va, GENet, ifs_G, ifs_GE)
    else:
        return([opt_L2, opt_Learning_Rate2, opt_lr, opt_l],opt_loss_train, opt_loss, opt_index_tr, opt_index_va, GENet)

