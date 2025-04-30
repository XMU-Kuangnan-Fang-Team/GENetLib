import torch
import pandas as pd

from GENetLib.scalar_mcp_l2train import scalar_mcp_l2train
from GENetLib.pre_data import pre_data2


'''G-E interaction analysis via neural network for scalar data input'''

pd.set_option('mode.chained_assignment', None)
def scalar_ge(y, G, E, ytype, num_hidden_layers, nodes_hidden_layer, num_epochs,
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
    # Define a function to identify important features based on a threshold t
    def important_features(tensor_, t):
        maxNum = max(abs(tensor_))
        resultPos = torch.where(abs(tensor_) > maxNum * t)[0].tolist()
        return resultPos
    # Model training of the neural network with MCP and L2 penaties
    ScalarMCP_L2trainRes = scalar_mcp_l2train(x_train, clinical_train, interaction_train, y_train,
                                              x_valid, clinical_valid, interaction_valid, y_valid,
                                              In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                              num_hidden_layers, nodes_hidden_layer, ytype, isfunc,
                                              learning_rate2, Lambda, learning_rate1, lambda2, 
                                              num_epochs, lambda1, plot, model, model_reg)
    # If t is provided, identify important features
    if threshold != None:
        ifs_G = important_features(ScalarMCP_L2trainRes[4].sparse1.weight.data, threshold)
        ifs_GE = important_features(ScalarMCP_L2trainRes[4].sparse2.weight.data, threshold)
    # Print performance metrics
    if ytype == 'Binary':
        print('Accuracy of train:', ScalarMCP_L2trainRes[0], 
              'Accuracy of test:', ScalarMCP_L2trainRes[1]) 
        print('AUC of train:', ScalarMCP_L2trainRes[2], 
              'AUC of test:', ScalarMCP_L2trainRes[3])
        if threshold != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Continuous':
        print('MSE of train:', ScalarMCP_L2trainRes[0].detach().numpy()[0], 
              'MSE of test:', ScalarMCP_L2trainRes[1].detach().numpy()[0]) 
        print('R2 of train:', ScalarMCP_L2trainRes[2], 
              'R2 of test:', ScalarMCP_L2trainRes[3])
        if threshold != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    elif ytype == 'Survival':
        print('Loss of train:', ScalarMCP_L2trainRes[0].detach().numpy()[0], 
              'Loss of test:', ScalarMCP_L2trainRes[1].detach().numpy()[0]) 
        print('C_index of train:', ScalarMCP_L2trainRes[2].detach().numpy(), 
              'C_index of test:', ScalarMCP_L2trainRes[3].detach().numpy())
        if threshold != None and important_feature == True:
            print('Important feature of gene:', ifs_G)
            print('Important feature of GE:', ifs_GE)
    if threshold != None:
        return(ScalarMCP_L2trainRes, ifs_G, ifs_GE)
    else:
        return(ScalarMCP_L2trainRes)
