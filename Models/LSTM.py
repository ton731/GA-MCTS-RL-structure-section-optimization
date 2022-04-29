from distutils.dir_util import create_tree
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




class LSTM_decay(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTM_decay, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        self.graph_encoder = MLP(self.input_dim, [], self.hidden_dim, act=True, dropout=False)
        self.lstmCellList = nn.ModuleList()
        for i in range(num_layers):
            self.lstmCellList.append(nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim))
        self.response_decoder = MLP(self.hidden_dim*2, [64], self.output_dim, act=True, dropout=False)


    def create_ground_motion_graph(self, gm, deacy_factor, graph_node, ptr):
        # add ground motion to original feature, then add decay_factor to hidden feature.
        bs = len(ptr)-1
        gm_graph = graph_node.clone()
        for b in range(bs):
            gm_graph[ptr[b]:ptr[b+1], 14:24] = gm[b*10 : (b+1)*10]
        gm_graph = self.graph_encoder(gm_graph)
        for b in range(bs):
            gm_graph[ptr[b]:ptr[b+1], -10:] = deacy_factor[b*10 : (b+1)*10]
        return gm_graph

    def set_hidden_state(self, graph, H):
        if H is None:
            H = self.graph_encoder(graph)
        return H

    def set_cell_state(self, graph, C):
        if C is None:
            C = self.graph_encoder(graph)
        return C

    def next_cell_input(self, H, gm, decay_factor, ptr):
        H_gm = H.clone()
        bs = len(ptr)-1
        for b in range(bs):
            H_gm[ptr[b]:ptr[b+1], -20:-10] = decay_factor[b*10 : (b+1)*10]
            H_gm[ptr[b]:ptr[b+1], -10:] = gm[b*10 : (b+1)*10]
        return H_gm

    def create_response(self, H, C):
        state = torch.cat([H, C], dim=1)
        node_out = self.response_decoder(state)
        return node_out


    def forward(self, gm, decay_factor, graph_node, edge_index, edge_attr, ptr, H_list, C_list):
        '''
        gm    : ground motion at time t [10*batch_size]
        graph : x[node_num(batch), feature_num]
        H     : hidden state at time t [node_num(batch), hidden_feature]
        C     : cell state at time t [node_num(batch), hidden_feature]
        '''

        X = self.create_ground_motion_graph(gm, decay_factor, graph_node, ptr)

        for i in range(self.num_layers):
            H_list[i] = self.set_hidden_state(graph_node, H_list[i])
            C_list[i] = self.set_hidden_state(graph_node, C_list[i])

        for i in range(self.num_layers):
            if i == 0:
                H_list[i], C_list[i] = self.lstmCellList[i](X, (H_list[i], C_list[i]))
            else:
                H_list[i], C_list[i] = self.lstmCellList[i](self.next_cell_input(H_list[i-1], gm, decay_factor, ptr),
                                                            (H_list[i], C_list[i]))
                # H_list[i], C_list[i] = self.lstmCellList[i](H_list[i-1], (H_list[i], C_list[i]))

        y = self.create_response(H_list[-1], C_list[-1])

        return H_list, C_list, y







class GraphLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, edge_attr_dim=15, node_sampler=None):
        super(GraphLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.edge_attr_dim = edge_attr_dim
        self.node_sampler = node_sampler


        # self.conv = torch_geometric.nn.GCNConv(input_dim, input_dim)
        self.conv = GraphNetwork_layer(input_dim=input_dim, output_dim=64, edge_attr_dim=edge_attr_dim)
        self.graph_encoder = MLP(64, [], self.hidden_dim, act=True, dropout=False)

        self.lstmCellList = nn.ModuleList()
        for i in range(num_layers):
            self.lstmCellList.append(nn.LSTMCell(input_size=hidden_dim, hidden_size=hidden_dim))
        self.response_decoder = MLP(self.hidden_dim*2, [64], self.output_dim, act=True, dropout=False)


    def create_node_convolution(self, node_conv, sample_ptr, graph):
        # strucutral graph ----message passing-----> multi-sectional graph ----sample-----> node_conv
        if node_conv is None:
            edge_ptr, sample_index_ptr = graph.edge_ptr, graph.sample_index_ptr

            # Set ptr for visualize (get graph from dataset instead of dataloader)
            try:    # batch loading
                ptr = graph.ptr
            except: # dataset loading
                ptr = [0, graph.x.shape[0]]
                edge_ptr = [edge_ptr]
                sample_index_ptr = [sample_index_ptr]


            sample_ptr = [0]
            node_conv = torch.Tensor().to("cuda")
            bs = len(ptr) - 1
            sample_start = 0
            edge_start = 0
            for b in range(bs):
                x = graph.x[ptr[b] : ptr[b+1], :]
                edge_index = graph.edge_index[:, edge_start : edge_start + edge_ptr[b]] - ptr[b]
                edge_attr = graph.edge_attr[edge_start : edge_start + edge_ptr[b]]
                all_conv = self.conv(x, edge_index, edge_attr)
                sample_index = [graph.sample_index[sample_start : sample_start + sample_index_ptr[b]].long() - ptr[b]]
                node_conv = torch.cat([node_conv, all_conv[sample_index]], dim=0)

                sample_start += sample_index_ptr[b]
                sample_ptr.append(sample_start)
                edge_start += edge_ptr[b]

        return node_conv, sample_ptr


    def create_ground_motion_graph(self, gm, node_conv, sample_ptr):
        # node_conv ----gm & decay factor----> gm graph ----graph encoder----> gm_graph
        ptr = sample_ptr
        bs = len(ptr)-1
        gm_graph = node_conv.clone()
        for b in range(bs):
            gm_graph[ptr[b]:ptr[b+1], -10:] = gm[b*10 : (b+1)*10]

        gm_graph = self.graph_encoder(gm_graph)
        return gm_graph

    def set_hidden_state(self, H, node_conv):
        # node_conv ----graph encoder----> H
        if H is None:
            H = self.graph_encoder(node_conv)
        return H

    def set_cell_state(self, C, node_conv):
        # node_conv ----graph encoder----> C
        if C is None:
            C = self.graph_encoder(node_conv)
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


    def forward(self, gm, node_conv, sample_ptr, graph, H_list, C_list):
        '''
        gm    : ground motion at time t [10*batch_size]
        graph : x[node_num(batch), feature_num]
        H     : hidden state at time t [node_num(batch), hidden_feature]
        C     : cell state at time t [node_num(batch), hidden_feature]
        '''

        node_conv, sample_ptr = self.create_node_convolution(node_conv, sample_ptr, graph)
        X = self.create_ground_motion_graph(gm, node_conv, sample_ptr)

        for i in range(self.num_layers):
            H_list[i] = self.set_hidden_state(H_list[i], node_conv)
            C_list[i] = self.set_hidden_state(C_list[i], node_conv)

        for i in range(self.num_layers):
            if i == 0:
                H_list[i], C_list[i] = self.lstmCellList[i](X, (H_list[i], C_list[i]))
            else:
                H_list[i], C_list[i] = self.lstmCellList[i](self.next_cell_input(H_list[i-1], gm, sample_ptr),
                                                            (H_list[i], C_list[i]))

        y = self.create_response(H_list[-1], C_list[-1])

        return node_conv, sample_ptr, H_list, C_list, y



