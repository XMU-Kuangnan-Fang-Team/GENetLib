import torch
import torch.optim as optim
import numpy as np
from torch.nn import BCELoss, MSELoss
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

from GENetLib.GE_Net import GE_Net
from GENetLib.survival_costfunc_cindex import neg_par_log_likelihood, c_index


'''Model training of the neural network with L2 penaty'''

dtype = torch.FloatTensor
def scalar_l2train(train_x, train_clinical, train_interaction, train_y,
                   eval_x, eval_clinical, eval_interaction, eval_y,
                   In_Nodes, Interaction_Nodes, Clinical_Nodes,
                   num_hidden_layers, nodes_hidden_layer, ytype, isfunc,
                   Learning_Rate2, L2, Num_Epochs, model_reg = None):
    # Initialize the GE_Net neural network
    net = GE_Net(In_Nodes, Interaction_Nodes, Clinical_Nodes, num_hidden_layers, nodes_hidden_layer, ytype, isfunc, model_reg)
    # Set up the optimizer
    opt = optim.Adam(net.parameters(), lr= Learning_Rate2, weight_decay = L2)
    # Training
    for epoch in range(Num_Epochs + 1):
        net.train()
        regularization_loss = 0
        pred = net(train_x, train_interaction, train_clinical)
        opt.zero_grad()
        # Calculate the loss function
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
        loss.backward() # Backpropagate the loss
        opt.step() # Update model parameters
        net_state_dict = net.state_dict()
        net.train()
        train_pred = net(train_x, train_interaction, train_clinical)
        # Calculate the training loss
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
        # Calculate the evaluation loss
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
    # Calculate and return performance metrics
    if ytype == 'Binary': # Accuracy and R2
        train_r2 = r2_score(train_y.detach().numpy(), train_pred.detach().numpy())
        eval_r2 = r2_score(eval_y.detach().numpy(), eval_pred.detach().numpy())
        train_y_pred_labels = np.where(np.array(train_pred.detach().numpy()) > 0.5, 1, 0)
        test_y_pred_labels = np.where(np.array(eval_pred.detach().numpy()) > 0.5, 1, 0)
        train_accuracy = accuracy_score(train_y.detach().numpy(), train_y_pred_labels)
        test_accuracy = accuracy_score(eval_y.detach().numpy(), test_y_pred_labels)
        return (train_accuracy, test_accuracy, train_r2, eval_r2, net)
    elif ytype == 'Continuous': # R2
        train_r2 = r2_score(train_y.detach().numpy(), train_pred.detach().numpy())
        eval_r2 = r2_score(eval_y.detach().numpy(), eval_pred.detach().numpy())
        return (loss, eval_loss, train_r2, eval_r2, net)
    elif ytype == 'Survival': # C_index
        train_cindex = c_index(train_pred, train_y[0], train_y[1])
        eval_cindex = c_index(eval_pred, eval_y[0], eval_y[1])
        return (loss, eval_loss, train_cindex, eval_cindex, net)
