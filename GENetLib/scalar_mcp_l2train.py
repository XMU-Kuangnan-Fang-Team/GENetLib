import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.nn import BCELoss, MSELoss
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score

from GENetLib.GE_Net import GE_Net
from GENetLib.survival_costfunc_cindex import neg_par_log_likelihood, c_index


'''Model training of the neural network with MCP and L2 penaties'''

dtype = torch.FloatTensor
def scalar_mcp_l2train(train_x, train_clinical, train_interaction, train_y,
                       eval_x, eval_clinical, eval_interaction, eval_y,
                       In_Nodes, Interaction_Nodes, Clinical_Nodes, 
                       num_hidden_layers, nodes_hidden_layer, ytype, isfunc,
                       Learning_Rate2, L2, Learning_Rate1, L, Num_Epochs, 
                       plot = True, model = None, model_reg = None):
    d = np.sqrt(Clinical_Nodes + 1)
    # Initialize the GE_Net neural network
    net = GE_Net(In_Nodes, Interaction_Nodes, Clinical_Nodes, num_hidden_layers, nodes_hidden_layer, ytype, isfunc, model_reg)
    if model != None:
        net.load_state_dict(model.state_dict(), strict=False)
    hidden_layers = getattr(net, 'hidden_layers')
    # Set up the optimizer for sparse layers and hidden layers
    opt = optim.Adam([
    {'params': net.sparse1.parameters(), 'weight_decay': 0, 'lr': Learning_Rate1 },
    {'params': net.sparse2.parameters(), 'weight_decay': 0, 'lr': Learning_Rate1 }] + [
    {'params': layer.parameters(), 'weight_decay': L2, 'lr': Learning_Rate2 }
    for layer in hidden_layers])
    loss_train_list = []
    loss_test_list = []
    # Training
    for epoch in range(Num_Epochs + 1):
        net.train()       
        regularization_loss = 0
        # MCP penalty
        b = torch.zeros([In_Nodes])
        for i in range(In_Nodes):
            temp = 0
            for j in range(Clinical_Nodes):
                temp += torch.abs(net.sparse2.weight.data[i + j * In_Nodes])
            b[i] = torch.abs(net.sparse1.weight.data[i]) + temp
        a = d * L - b/torch.tensor([3])
        zero = torch.zeros([In_Nodes])
        Pb = torch.where(a<0, zero, a)
        w1 = Pb / (2 * b ** 2)
        for param in net.sparse1.parameters():
            regularization_loss += torch.sum(param ** 2 * torch.abs(param.data) * w1)
        bInter = torch.Tensor(w1.tolist() * Clinical_Nodes)
        for param in net.sparse2.parameters():
            a = L - torch.abs(param.data)/torch.tensor([3])
            zero = torch.zeros(Interaction_Nodes)
            Pbeta = torch.where(a<0, zero, a)
            regularization_loss += torch.sum(param ** 2 * (torch.abs(param.data) * bInter + Pbeta / (2 * torch.abs(param.data))))
        pred = net(train_x, train_interaction, train_clinical)
        opt.zero_grad()
        # Calculate the loss function
        if ytype == 'Survival':
            loss = neg_par_log_likelihood(pred, train_y[0], train_y[1]) + regularization_loss
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
            train_loss = loss_fn(train_pred, train_y[0], train_y[1]).view(1,) + regularization_loss
            loss_train_list.append(train_loss.detach().numpy())
        elif ytype == 'Binary':
            loss_fn = BCELoss()
            train_loss = loss_fn(train_pred, train_y).view(1,) + regularization_loss
            loss_train_list.append(train_loss.detach().numpy())
        elif ytype == 'Continuous':
            loss_fn = MSELoss()
            train_loss = loss_fn(train_pred, train_y).view(1,) + regularization_loss
            loss_train_list.append(train_loss.detach().numpy())
        else:
            raise ValueError('Invalid ytype')
        net.eval()
        eval_pred = net(eval_x, eval_interaction, eval_clinical)
        # Calculate the evaluation loss
        if ytype == 'Survival':
            loss_fn = neg_par_log_likelihood
            eval_loss = loss_fn(eval_pred, eval_y[0], eval_y[1]).view(1,) + regularization_loss
            loss_test_list.append(eval_loss.detach().numpy())
        elif ytype == 'Binary':
            loss_fn = BCELoss()
            eval_loss = loss_fn(eval_pred, eval_y).view(1,) + regularization_loss
            loss_test_list.append(eval_loss.detach().numpy())
        elif ytype == 'Continuous':
            loss_fn = MSELoss()
            eval_loss = loss_fn(eval_pred, eval_y).view(1,) + regularization_loss
            loss_test_list.append(eval_loss.detach().numpy())
        else:
            raise ValueError('Invalid ytype')
    # Plot training loss versus evaluation loss
    if plot == True:
        plt.plot(np.log(np.array(loss_train_list)), label='train')
        plt.plot(np.log(np.array(loss_test_list)), label='test')
        plt.legend(prop = {'size':18})
        plt.show()
    # Calculate and return performance metrics
    if ytype == 'Binary': # Accuracy and R2
        train_y_pred_labels = np.where(np.array(train_pred.detach().numpy()) > 0.5, 1, 0)
        test_y_pred_labels = np.where(np.array(eval_pred.detach().numpy()) > 0.5, 1, 0)
        train_accuracy = accuracy_score(train_y.detach().numpy(), train_y_pred_labels)
        test_accuracy = accuracy_score(eval_y.detach().numpy(), test_y_pred_labels)
        train_auc = roc_auc_score(train_y.detach().numpy(), np.array(train_pred.detach().numpy()))
        test_auc = roc_auc_score(eval_y.detach().numpy(), np.array(eval_pred.detach().numpy()))
        return (train_accuracy, test_accuracy, train_auc, test_auc, net)
    elif ytype == 'Continuous': # R2
        train_r2 = r2_score(train_y.detach().numpy(), train_pred.detach().numpy())
        eval_r2 = r2_score(eval_y.detach().numpy(), eval_pred.detach().numpy())
        return (train_loss, eval_loss, train_r2, eval_r2, net)
    elif ytype == 'Survival': # C_index
        train_cindex = c_index(train_pred, train_y[0], train_y[1])
        eval_cindex = c_index(eval_pred, eval_y[0], eval_y[1])
        return (train_loss, eval_loss, train_cindex, eval_cindex, net)
