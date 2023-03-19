import torch
import torch.nn as nn
from .layers import *


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.graph_encoder = MLP(self.input_dim, [], self.hidden_dim, act=True, dropout=False)
        self.lstmCellList = nn.ModuleList()
        for i in range(num_layers):
            self.lstmCellList.append(nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim))
        self.response_decoder = MLP(self.hidden_dim*2, [64], self.output_dim, act=True, dropout=False)


    def create_ground_motion_graph(self, gm, graph_node, ptr):
        bs = len(ptr)-1
        gm_graph = graph_node.clone()
        for b in range(bs):
            gm_graph[ptr[b]:ptr[b+1], -10:] = gm[b*10 : (b+1)*10]
        gm_graph = self.graph_encoder(gm_graph)
        return gm_graph

    def set_hidden_state(self, graph, H):
        if H is None:
            H = self.graph_encoder(graph)
        return H

    def set_cell_state(self, graph, C):
        if C is None:
            C = self.graph_encoder(graph)
        return C

    def next_cell_input(self, H, gm, ptr):
        H_gm = H.clone()
        bs = len(ptr)-1
        for b in range(bs):
            H_gm[ptr[b]:ptr[b+1], -10:] = gm[b*10 : (b+1)*10]
        return H_gm

    def create_response(self, H, C):
        state = torch.cat([H, C], dim=1)
        node_out = self.response_decoder(state)
        return node_out


    def forward(self, gm, graph_node, edge_index, edge_attr, ptr, H_list, C_list):
        '''
        gm    : ground motion at time t [10*batch_size]
        graph : x[node_num(batch), feature_num]
        H     : hidden state at time t [node_num(batch), hidden_feature]
        C     : cell state at time t [node_num(batch), hidden_feature]
        '''

        X = self.create_ground_motion_graph(gm, graph_node, ptr)

        for i in range(self.num_layers):
            H_list[i] = self.set_hidden_state(graph_node, H_list[i])
            C_list[i] = self.set_hidden_state(graph_node, C_list[i])

        for i in range(self.num_layers):
            if i == 0:
                H_list[i], C_list[i] = self.lstmCellList[i](X, (H_list[i], C_list[i]))
            else:
                H_list[i], C_list[i] = self.lstmCellList[i](self.next_cell_input(H_list[i-1], gm, ptr),
                                                            (H_list[i], C_list[i]))
                # H_list[i], C_list[i] = self.lstmCellList[i](H_list[i-1], (H_list[i], C_list[i]))

        y = self.create_response(H_list[-1], C_list[-1])

        return H_list, C_list, y
