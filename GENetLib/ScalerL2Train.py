from GENet import GE_Net
from Survival_CostFunc_CIndex import neg_par_log_likelihood, c_index
import torch
import torch.optim as optim
import numpy as np
from torch.nn import BCELoss, MSELoss
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

'''
train_y/evel_y: for cox model, y is a list, first column is ytime, second column is yevent
'''

dtype = torch.FloatTensor
def ScalerL2train(train_x, train_clinical, train_interaction, train_y,
                  eval_x, eval_clinical, eval_interaction, eval_y,
                  In_Nodes, Interaction_Nodes, Clinical_Nodes,
                  num_hidden_layers, nodes_hidden_layer, ytype, issnp,
                  Learning_Rate2, L2, Num_Epochs, model_reg = None):
    net = GE_Net(In_Nodes, Interaction_Nodes, Clinical_Nodes, num_hidden_layers, nodes_hidden_layer, ytype, issnp, model_reg)
    ###optimizer
    # opt = optim.Adam(net.parameters(), lr=Learning_Rate, weight_decay = L2)
    opt = optim.Adam(net.parameters(), lr= Learning_Rate2, weight_decay = L2)
    for epoch in range(Num_Epochs + 1):
        net.train()
        regularization_loss = 0
        pred = net(train_x, train_interaction, train_clinical)
        opt.zero_grad()  ###reset gradients to zeros
        if ytype == 'Survival':
            loss_fn = neg_par_log_likelihood
            loss = loss_fn(pred, train_y[0], train_y[1]) + regularization_loss
        elif ytype == 'Binary':
            loss_fn = BCELoss()
            loss = loss_fn(pred, train_y) + regularization_loss
        elif ytype == 'Continuous':
            loss_fn = MSELoss()
            loss = loss_fn(pred, train_y) + regularization_loss
        else:
            raise ValueError('Invalid ytype')
        loss.backward()  ###calculate gradients
        opt.step()  ###update weights and biases
        ###serializing net
        net_state_dict = net.state_dict()
        net.train()
        train_pred = net(train_x, train_interaction, train_clinical)
        if ytype == 'Survival':
            loss_fn = neg_par_log_likelihood
            loss = loss_fn(train_pred, train_y[0], train_y[1]) + regularization_loss
        elif ytype == 'Binary':
            loss_fn = BCELoss()
            loss = loss_fn(train_pred, train_y) + regularization_loss
        elif ytype == 'Continuous':
            loss_fn = MSELoss()
            loss = loss_fn(train_pred, train_y) + regularization_loss
        else:
            raise ValueError('Invalid ytype')
        net.eval()
        eval_pred = net(eval_x, eval_interaction, eval_clinical)
        if ytype == 'Survival':
            loss_fn = neg_par_log_likelihood
            eval_loss = loss_fn(eval_pred, eval_y[0], eval_y[1]).view(1,) + regularization_loss
        elif ytype == 'Binary':
            loss_fn = BCELoss()
            eval_loss = loss_fn(eval_pred, eval_y).view(1,) + regularization_loss
        elif ytype == 'Continuous':
            loss_fn = MSELoss()
            eval_loss = loss_fn(eval_pred, eval_y).view(1,) + regularization_loss
        else:
            raise ValueError('Invalid ytype')
    if ytype == 'Binary':
        train_r2 = r2_score(train_y.detach().numpy(), train_pred.detach().numpy())
        eval_r2 = r2_score(eval_y.detach().numpy(), eval_pred.detach().numpy())
        train_y_pred_labels = np.where(np.array(train_pred.detach().numpy()) > 0.5, 1, 0)
        test_y_pred_labels = np.where(np.array(eval_pred.detach().numpy()) > 0.5, 1, 0)
        train_accuracy = accuracy_score(train_y.detach().numpy(), train_y_pred_labels)
        test_accuracy = accuracy_score(eval_y.detach().numpy(), test_y_pred_labels)
        return (train_accuracy, test_accuracy, train_r2, eval_r2, net)
    elif ytype == 'Continuous':
        train_r2 = r2_score(train_y.detach().numpy(), train_pred.detach().numpy())
        eval_r2 = r2_score(eval_y.detach().numpy(), eval_pred.detach().numpy())
        return (loss, eval_loss, train_r2, eval_r2, net)
    elif ytype == 'Survival':
        train_cindex = c_index(train_pred, train_y[0], train_y[1])
        eval_cindex = c_index(eval_pred, eval_y[0], eval_y[1])
        return (loss, eval_loss, train_cindex, eval_cindex, net)


'''
test

from SimDataScaler import SimDataScaler
from PreData1 import PreData1
ytype = 'Survival'
scaler_survival_linear = SimDataScaler(0.25, 0.3, 500, 5, 1500, 2, ytype, 30)
x_train, y_train, clinical_train, interaction_train,\
x_valid, y_valid, clinical_valid, interaction_valid,\
x_test, y_test, clinical_test, interaction_test = PreData1(scaler_survival_linear, 500, 5, 2500, ytype, split_type = 1, ratio = [3, 1, 1])
In_Nodes = 500
Interaction_Nodes = 2500 
Clinical_Nodes = 5
num_hidden_layers = 2
nodes_hidden_layer = [1000, 100]
Learning_Rate2 = 0.005
L2 = 0.1
Num_Epochs = 100
ScalerL2trainRes = ScalerL2train(x_train, clinical_train, interaction_train, y_train,
                                 x_valid, clinical_valid, interaction_valid, y_valid,
                                 In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                                 num_hidden_layers, nodes_hidden_layer, ytype,
                                 Learning_Rate2, L2, Num_Epochs)

print(ScalerL2trainRes)
'''