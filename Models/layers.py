import torch
import torch.nn as nn
import torch.nn.functional as F





class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, act=False, dropout=False, p=0.5, **kwargs):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.act = act
        self.p = p
        concat_dim = [input_dim] + list(hidden_dim) + [output_dim]

        self.module_list = nn.ModuleList()
        for i in range(len(concat_dim)-1):
            self.module_list.append(nn.Linear(concat_dim[i], concat_dim[i+1]))
    
    def forward(self, x):
        for i, module in enumerate(self.module_list):
            x = module(x)
            if self.act and i != len(self.module_list)-1:
                x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, p=self.p, training=self.training)
        return x

