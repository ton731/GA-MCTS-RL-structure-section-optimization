import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import init
from torch.nn.utils import weight_norm
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import math



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

        

class GraphNetwork_layer(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_attr_dim=15, message_dim=None, aggr='add', act=False):
        super(GraphNetwork_layer, self).__init__()
        self.aggr = aggr
        self.act = act
        message_dim = input_dim if message_dim is None else message_dim

        self.messageMLP = MLP(input_dim * 2 + edge_attr_dim, [], message_dim, act=self.act, dropout=False)
        self.outputMLP = MLP(input_dim + message_dim, [], output_dim, act=self.act, dropout=False)
        

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, messageMLP=self.messageMLP, outputMLP=self.outputMLP)

    def message(self, x_i, x_j, edge_attr, messageMLP):
        return messageMLP(torch.cat((x_i, x_j, edge_attr), dim=-1))

    def update(self, aggr_out, x, outputMLP):
        return outputMLP(torch.cat((x, aggr_out), dim=-1))




class SectionMessagePassingLayer(MessagePassing):
    def __init__(self, edge_attr_dim=15, gm_dim=10, aggr='add'):
        super(SectionMessagePassingLayer, self).__init__()
        '''
        Edge attr dim = 15
        Input  x: [node_num, feature_num]
        Output x: [node_num, feature_num + edge_attr]
        '''
        self.aggr = aggr
        self.edge_attr_dim = edge_attr_dim
        self.gm_dim = gm_dim

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        gm_space = torch.zeros((x.shape[0], self.gm_dim))
        return torch.cat([x[:, :-self.gm_dim], aggr_out, gm_space], dim=1)





'''
class NodeSampleLayer(torch.nn.Module):
    def __init__(self, norm_dict, denormalize_x, reduce_sample=True):
        super(NodeSampleLayer, self).__init__()
        self.norm_dict = norm_dict
        self.reduce_sample = reduce_sample

    def forward(self, node_conv):
        graph = node_conv.clone()
        # Sampled the node on the zigzag path in each story.  
        x_grid_num, y_grid_num, z_grid_num = graph.grid_num.numpy().astype(int)
        x_grid_num, z_grid_num = x_grid_num - 1, z_grid_num - 1
        sampled_xz_coord = []
        if_x_more = True if x_grid_num >= z_grid_num else False
        x = 0
        z = 0
        increase = False
        while((x <= x_grid_num) if if_x_more else (z <= z_grid_num)):
            sampled_xz_coord.append([x, z])

            # Check if need to change direction of zigzag.
            if (if_x_more and (z == 0 or z == z_grid_num)) or (not if_x_more and (x == 0 or x == x_grid_num)):
                increase = not increase

            # Update the zigzag pointer
            if increase:
                x += 1
                z += 1
            else:
                if if_x_more:
                    x += 1
                    z -= 1
                else:
                    x -= 1
                    z += 1

        sampled_node_index = []
        topology = self.denormalize_x(graph.x[:, :6], self.norm_dict)
        for index in range(graph.x.shape[0]):
            # Check if node's x, z grid coord is in the zigzag path.
            x_grid, y_grid, z_grid = topology[index, 3:6].cpu().numpy()
            if  [x_grid, z_grid] in sampled_xz_coord:
                sampled_node_index.append(index)

        # Reduce the number of sampled node for more data number.
        if self.reduce_sample:
            sampled_node_index = [index for i, index in enumerate(sampled_node_index) if i%2==0]

        graph.x = graph.x[sampled_node_index]
        graph.y = graph.y[sampled_node_index]

        return graph, sampled_node_index

'''

















class Edge_GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, bias=True, aggr="add"):
        super(Edge_GCNConv, self).__init__(aggr=aggr)

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels))

        self.edge_dim = edge_dim
        self.edge_update = torch.nn.Parameter(torch.Tensor(out_channels + edge_dim, out_channels))  # new

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edge_update)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):

        x = torch.matmul(x, self.weight)
        edge_weight = torch.ones((edge_index.size(1),),
                                 dtype=x.dtype,
                                 device=edge_index.device)  
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 1, x.size(0)) 

        self_loop_edges = torch.zeros(x.size(0), edge_attr.size(1)).to(edge_index.device)
        edge_attr = torch.cat([edge_attr, self_loop_edges], dim=0)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5) 
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr, norm):
        x_j = torch.cat([x_j, edge_attr], dim=-1)
        return norm.view(-1, 1) * x_j 

    def update(self, aggr_out):
        aggr_out = torch.mm(aggr_out, self.edge_update)
        if self.bias is not None:
            return aggr_out + self.bias
        else:
            return aggr_out


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


