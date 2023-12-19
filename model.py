import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.utils import degree
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, global_sort_pool,GATConv,SAGPooling,LayerNorm,GraphConv,global_add_pool
from torch_geometric.utils import dropout_adj

from util_f import *
from torch.nn.modules.container import ModuleList

import torch
from torch_geometric.nn import GCNConv


class attention_score(torch.nn.Module):
    def __init__(self, in_channels, Conv=GCNConv):
        super(attention_score, self).__init__()
        self.in_channels = in_channels
        self.score_layer = Conv(in_channels, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        score = self.score_layer(x, edge_index)

        return score
class Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats, final_out_feats):
        super().__init__()
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = head_out_feats
        self.conv = GATConv(in_features, head_out_feats, n_heads)
        self.layer_norm = True
        self.batch_norm = False
        self.residual = True
        self.layer_norm1_h = nn.LayerNorm(64)
        self.layer_norm2_h = nn.LayerNorm(64)
        self.batch_norm1_h = nn.BatchNorm1d(64)
        self.batch_norm2_h = nn.BatchNorm1d(64)
        self.FFN_h_layer1 = nn.Linear(64, 64 * 2)
        self.FFN_h_layer2 = nn.Linear(64 * 2, 64)

    def forward(self, data, e):
        data = self.conv(data, e)
        h1 = data
        if self.layer_norm:
            data = self.layer_norm1_h(data)
        if self.batch_norm:
            data = self.batch_norm1_h(data)
        data = self.FFN_h_layer1(data)
        data = F.relu(data)
        data = F.dropout(data,0,training=True)
        data = self.FFN_h_layer2(data)
        if self.residual:
            data = h1 + data
        if self.layer_norm:
            data = self.layer_norm2_h(data)
        if self.batch_norm:
            data = self.batch_norm2_h(data)
        return data

class DVGEDR(torch.nn.Module):
    def __init__(self, dataset1,dataset2, gconv=GCNConv, latent_dim=[64, 64,64], k1=30, k2=30,
                 dropout_n=0.4, dropout_e=0.1,  force_undirected=False, device=torch.device('cuda:1'),in_features=18):
        super(DVGEDR, self).__init__()
        self.device = device
        self.dropout_n = dropout_n
        self.dropout_e = dropout_e
        self.force_undirected = force_undirected
        self.score1 = attention_score(latent_dim[0])
        self.score2 = attention_score(latent_dim[1])
        self.score3 = attention_score(latent_dim[2])
        self.layer_norm = False
        self.batch_norm = True
        self.residual = True
        self.conv1 = gconv(dataset1.num_features, latent_dim[0])
        self.conv2 = gconv(latent_dim[0], latent_dim[1])
        self.conv3 = gconv(latent_dim[1], latent_dim[2])
        self.gat1 = GATConv(in_features, 32, 2)
        self.gat2 = GATConv(64,32,2)
        self.in_features = in_features
        self.hidd_dim = 64
        self.n_blocks = len([2,2])

        self.initial_norm = LayerNorm(self.in_features)
        self.blocks = []
        self.net_norms = ModuleList()
        for i, (head_out_feats, n_heads) in enumerate(zip([32,32,32], [2,2,2])):
            block = Block(n_heads, in_features, head_out_feats, final_out_feats=self.hidd_dim)
            self.add_module(f"block{i}", block)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(head_out_feats * n_heads))
            in_features = head_out_feats * n_heads

        if k1 < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset1])
            k1 = node_nums[int(math.ceil(k1 * len(node_nums)))-1]
            k1 = max(10, k1)  # no smaller than 10
        if k2 < 1:  # transform percentile to number
            node_nums = sorted([g.num_nodes for g in dataset2])
            k2 = node_nums[int(math.ceil(k2 * len(node_nums)))-1]
            k2 = max(10, k2)
        self.k1 = int(k1)
        self.k2 = int(k2)
        conv1d_channels = [16, 32]
        self.total_latent_dim = sum(latent_dim)
        conv1d_kws = [self.total_latent_dim, 5]
        self.conv1d_params1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = nn.MaxPool1d(2, 2)

        self.conv1d_params2 = Conv1d(conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1)
        dense_dim1 = int((k1 - 2) / 2 + 1)
        dense_dim2 = int((k2 - 2) / 2 + 1)
        self.dense_dim1 = (dense_dim1 - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.dense_dim2 = (dense_dim2 - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(self.dense_dim1, 128)
        self.lin12 = Linear(self.dense_dim2, 128)
        self.lin2 = Linear(128, 1)
        self.lin3 = Linear(192,128)

        self.w_i = nn.Linear(128, 128)
        self.prj_i = nn.Linear(128, 128)
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.conv1d_params1.reset_parameters()
        self.conv1d_params2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
    def cross_attention(self,drug,batch,g_align,test,epoch):
        scores = (drug * self.prj_i(g_align)).sum(-1)
        scores = softmax(scores, batch, dim=0)
        final = global_add_pool(drug * scores.unsqueeze(-1), batch)
        return final
    def forward(self, data1,data2,epoch,test):
        x1, edge_index1, batch1 = data1.x, data1.edge_index, data1.batch
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
        edge_index1, _ = dropout_adj(
            edge_index1, p=self.dropout_e,
            force_undirected=self.force_undirected, num_nodes=len(x1),
            training=self.training
        )

        edge_index2, _ = dropout_adj(
            edge_index2, p=self.dropout_e,
            force_undirected=self.force_undirected, num_nodes=len(x2),
            training=self.training
        )
        h_graph1 = []
        h_graph2 = []
        x1 = self.initial_norm(x1, batch1)
        x2 = self.initial_norm(x2, batch2)
        for i, block in enumerate(self.blocks):
            out1 = block(x1, edge_index1)
            out2 = block(x2, edge_index2)
            h_graph1.append(out1)
            h_graph2.append(out2)
            x1 = out1
            x2 = out2

        attention_score11 = self.score1(torch.relu(h_graph1[0]), edge_index1)
        x11 = torch.mul(attention_score11, h_graph1[0])
        attention_score12 = self.score1(torch.relu(h_graph2[0]), edge_index2)
        x12 = torch.mul(attention_score12, h_graph2[0])

        attention_score21 = self.score2(torch.relu(h_graph1[1]), edge_index1)
        x21 = torch.mul(attention_score21, h_graph1[1])
        attention_score22 = self.score2(torch.relu(h_graph2[1]), edge_index2)
        x22 = torch.mul(attention_score22, h_graph2[1])

        attention_score31 = self.score3(torch.relu(h_graph1[2]), edge_index1)
        x31 = torch.mul(attention_score31, h_graph1[2])
        attention_score32 = self.score3(torch.relu(h_graph2[2]), edge_index2)
        x32 = torch.mul(attention_score32, h_graph2[2])

        X1 = [x11, x21, x31]
        X2 = [x12, x22, x32]
        concat_states1 = torch.cat(X1, 1)
        concat_states2 = torch.cat(X2, 1)

        x1 = global_sort_pool(concat_states1, batch1, self.k1)  # batch * (k*hidden)
        x1 = x1.unsqueeze(1)  # batch * 1 * (k*hidden)
        x1 = F.relu(self.conv1d_params1(x1))
        x1 = self.maxpool1d(x1)
        x1 = F.relu(self.conv1d_params2(x1))
        x1 = x1.view(len(x1), -1)  # flatten

        x2 = global_sort_pool(concat_states2, batch2, self.k2)  # batch * (k*hidden)
        x2 = x2.unsqueeze(1)  # batch * 1 * (k*hidden)
        x2 = F.relu(self.conv1d_params1(x2))
        x2 = self.maxpool1d(x2)
        x2 = F.relu(self.conv1d_params2(x2))
        x2 = x2.view(len(x2), -1)  # flatten

        x1 = F.relu(self.lin1(x1))
        x1 = F.dropout(x1, p=self.dropout_n, training=self.training)
        x2 = F.relu(self.lin12(x2))
        x2 = F.dropout(x2, p=self.dropout_n, training=self.training)

        concat_states1 = self.lin3(concat_states1)
        concat_states2 = self.lin3(concat_states2)
        g_align1 = x1.repeat_interleave(degree(data1.batch, dtype=data1.batch.dtype), dim=0)
        x1 = self.cross_attention(concat_states1, data1.batch, g_align1,test,epoch)

        g_align2 = x2.repeat_interleave(degree(data2.batch, dtype=data2.batch.dtype), dim=0)
        x2 = self.cross_attention(concat_states2, data2.batch, g_align2)

        #x = torch.cat([x1, x2], dim=-1)
        x = 0.9 * x1 + 0.1 * x2
        x = self.lin2(x)

        return x[:, 0]
