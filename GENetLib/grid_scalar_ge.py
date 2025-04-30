import pandas as pd
import torch

from GENetLib.pre_data import pre_data2
from GENetLib.scalar_l2train import scalar_l2train
from GENetLib.scalar_mcp_l2train import scalar_mcp_l2train


'''Grid search for scalar_ge'''

pd.set_option('mode.chained_assignment', None)
def grid_scalar_ge(y, G, E, ytype, num_hidden_layers, nodes_hidden_layer, num_epochs,
                   learning_rate1, learning_rate2, lambda1 = None, lambda2 = None, Lambda = None,
                   threshold = None, model = None, split_type = 0, ratio = [7, 3], 
                   important_feature = True, plot = True, model_reg = None, isfunc = False):
    # Define dimensions for gene, environment and gene-environment features
    if lambda2 == None:
        print('Please enter lambda2')
    if Lambda == None:
        print('Please enter Lambda')
    In_Nodes = G.shape[1]
    Clinical_Nodes = E.shape[1]
    Interaction_Nodes = G.shape[1] * E.shape[1]
    x = G
    clinical = E
    interaction = None
    # Split data into training and validation(test) sets
    if split_type == 1:
        x_train, y_train, clinical_train, interaction_train,\
        x_valid, y_valid, clinical_valid, interaction_valid,\
        x_test, y_test, clinical_test, interaction_test = pre_data2(y, x, clinical, interaction, ytype, split_type, ratio)
    elif split_type == 0:
        x_train, y_train, clinical_train, interaction_train,\
        x_valid, y_valid, clinical_valid, interaction_valid = pre_data2(y, x, clinical, interaction, ytype, split_type, ratio)
    # Grid search for learning rate of L2 penalty
    opt_loss = torch.Tensor([float("Inf")])
    for lr2 in learning_rate2:
        for l2 in Lambda:
            loss_train, loss_valid, index_tr, index_va, model = scalar_l2train(x_train, clinical_train, interaction_train, y_train,
                                                                               x_valid, clinical_valid, interaction_valid, y_valid,
                                                                               In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                                                               num_hidden_layers, nodes_hidden_layer, ytype, isfunc,
                                                                               lr2, l2, num_epochs, model_reg)
            if loss_valid < opt_loss:
                opt_L2 = l2
                opt_Learning_Rate2 = lr2
                best_model = model
                opt_loss = loss_valid
    # Grid search for learning rate of MCP penalty
    opt_loss = torch.Tensor([float("Inf")])
    GENet = None
    for l in lambda2:
        for lrMCP in learning_rate1:
            loss_train, loss_valid, index_tr, index_va, MCPNet = scalar_mcp_l2train(x_train, clinical_train, interaction_train, y_train,
                                                                                    x_valid, clinical_valid, interaction_valid, y_valid,
                                                                                    In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                                                                    num_hidden_layers, nodes_hidden_layer, ytype, isfunc,
                                                                                    opt_Learning_Rate2, opt_L2, lrMCP, l, num_epochs, lambda1, 
                                                                                    plot, best_model, model_reg)
            if loss_valid < opt_loss:
                opt_loss = loss_valid
                GENet = MCPNet
                opt_l = l
                opt_lr = lrMCP
                opt_loss_train = loss_train
                opt_index_tr = index_tr
                opt_index_va = index_va
    # Define a function to identify important features based on a threshold t
    def important_features(tensor_, t):
        maxNum = max(abs(tensor_))
        resultPos = torch.where(abs(tensor_) > maxNum * t)[0].tolist()
        return resultPos
    if threshold != None:
        tensor1 = GENet.sparse1.weight.data
        tensor2 = GENet.sparse2.weight.data
        ifs_G = important_features(tensor1, threshold)
        ifs_GE = important_features(tensor2, threshold)
    # Print performance metrics
    if ytype == 'Binary':
        print('opt_index: L2:', opt_L2, "LR2:", opt_Learning_Rate2, "LR_MCP:", opt_lr, 'L:', opt_l)
        print('Accuracy of train:', opt_loss_train, 'Accuracy of test:', opt_loss) 
        print('AUC of train:', opt_index_tr, 'AUC of test:', opt_index_va)
        if threshold != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Continuous':
        print('opt_index: L2:', opt_L2, "LR2:", opt_Learning_Rate2, "LR_MCP:", opt_lr, 'L:', opt_l)
        print('MSE of train:', opt_loss_train.detach().numpy()[0], 'MSE of test:', opt_loss.detach().numpy()[0]) 
        print('R2 of train:', opt_index_tr, 'R2 of test:', opt_index_va)
        if threshold != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Survival':
        print('opt_index: L2:', opt_L2, "LR2:", opt_Learning_Rate2, "LR_MCP:", opt_lr, 'L:', opt_l)
        print('Loss of train:', opt_loss_train.detach().numpy(), 
              'Loss of test:', opt_loss.detach().numpy()) 
        print('C_index of train:', opt_index_tr.detach().numpy(), 
              'C_index of test:', opt_index_va.detach().numpy())
        if threshold != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    if threshold != None:
        return([opt_L2, opt_Learning_Rate2, opt_lr, opt_l],opt_loss_train, opt_loss, opt_index_tr, opt_index_va, GENet, ifs_G, ifs_GE)
    else:
        return([opt_L2, opt_Learning_Rate2, opt_lr, opt_l],opt_loss_train, opt_loss, opt_index_tr, opt_index_va, GENet)

