import torch
import torch.nn as nn


class weight_sparse(nn.Module):    
    def __init__(self, in_features):
        super(weight_sparse, self).__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.Tensor(self.in_features))
        self.weight.data.uniform_(0, 1)
    def forward(self, input):
        x = input * self.weight
        return x

class weight_sparse1(nn.Module):    
    def __init__(self, in_nodes, model_reg):
        super(weight_sparse1, self).__init__()
        self.in_nodes = in_nodes
        self.weight = nn.Parameter(torch.Tensor(self.in_nodes))
        coef = model_reg.coef_.reshape(-1)
        self.weight.data = torch.from_numpy(coef[:self.in_nodes]).float()
    def forward(self, input):
        x = input * self.weight
        return x

class weight_sparse2(nn.Module):    
    def __init__(self, in_nodes, clinical_nodes, model_reg):
        super(weight_sparse2, self).__init__()
        self.in_nodes = in_nodes
        self.clinical_nodes = clinical_nodes
        self.weight = nn.Parameter(torch.Tensor(self.in_nodes*self.clinical_nodes))
        coef = model_reg.coef_.reshape(-1)
        self.weight.data = torch.from_numpy(coef[self.in_nodes:self.in_nodes*self.clinical_nodes+self.in_nodes]).float()
    def forward(self, input):
        x = input * self.weight
        return x

class GE_Net(nn.Module):
    def __init__(self, In_Nodes, Interaction_Nodes, Clinical_Nodes, num_hidden_layers, nodes_hidden_layer, ytype, issnp, model_reg):
        super(GE_Net, self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.ytype = ytype
        if issnp == True:
            self.sparse1 = weight_sparse1(In_Nodes, model_reg)
            self.sparse2 = weight_sparse2(In_Nodes, Clinical_Nodes, model_reg)
        else:
            self.sparse1 = weight_sparse(In_Nodes)
            self.sparse2 = weight_sparse(Interaction_Nodes)
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(In_Nodes + Interaction_Nodes + Clinical_Nodes, nodes_hidden_layer[0]))
        for i in range(1,num_hidden_layers):
            self.hidden_layers.append(nn.Linear(nodes_hidden_layer[i-1], nodes_hidden_layer[i]))
        self.hidden_layers.append(nn.Linear(nodes_hidden_layer[-1], 1, bias = False))
        self.hidden_layers[-1].weight.data.uniform_(-0.001, 0.001)
    
    def forward(self, x_1, x_2, x_3):
        x_1 = self.sparse1(x_1)
        x_2 = self.sparse2(x_2)
        x = torch.cat((x_1, x_2),1)
        x = torch.cat((x, x_3),1)   
        for i in range(len(self.hidden_layers) - 1):
            x = self.relu(self.hidden_layers[i](x))
        if self.ytype == 'Binary':
            lin_pred = self.sigmoid(self.hidden_layers[-1](x))
        else:
            lin_pred = self.hidden_layers[-1](x)
        return lin_pred
