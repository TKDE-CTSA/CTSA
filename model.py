import torch.nn as nn
import torch.nn.functional as F
import torch
import utils
import math
import numpy as np
from layers import GraphConvolution, GraphConvolution_Sparse


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
        # return F.log_softmax(x, dim=1)


class DGAM(nn.Module):
    def __init__(self,
                 node_num,
                 feat_num,
                 hidden_size,
                 emb_size,
                 dropout,
                 num_layers=1,
                 gamma=3.0,
                 num_sample=5,
                 n_head=8,
                 dim_feedforward=512,
                 sample_nodes=None):
        super(DGAM, self).__init__()
        self.node_num = node_num
        self.hidden_size = hidden_size
        self.feat_num = feat_num
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.gamma = gamma
        self.n_head = n_head
        self.num_sample = num_sample
        self.dim_feedforward = dim_feedforward
        self.sample_nodes = sample_nodes
        self.pos_encoder = PositionalEncoding(hidden_size, feat_num, hidden_size, dropout=self.dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                        nhead=self.n_head,
                                                        dim_feedforward=self.dim_feedforward,
                                                        dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.gc_1 = GraphConvolution(self.feat_num, self.hidden_size)
        self.gc_2 = GraphConvolution(self.hidden_size, self.emb_size)

        # self.gc_1 = GraphConvolution_Sparse(self.node_num, self.hidden_size)
        # self.gc_2 = GraphConvolution_Sparse(self.hidden_size, self.emb_size)

    def forward(self, features, adjs, train):
        #
        train_num = len(train)

        embs_l1 = F.relu(self.gc_1(features, adjs))

        embs_l1_1 = embs_l1[:, train_num:, :]
        embs_l2_2 = embs_l1[:, :train_num, :]

        embs_trans_1 = self.transformer_encoder(embs_l1_1)

        features = torch.cat((embs_trans_1, embs_l2_2), dim=1)
        features = F.dropout(features, self.dropout, training=self.training)

        self.embs = self.gc_2(features, adjs)

        return self.embs

    def calculate_loss(self, train, time_1, time_2):

        train_num = len(train)
        embs1 = self.embs[time_1, :, :]
        embs2 = self.embs[time_2, :, :]
        idx_train = train[:, 0]

        criterion = nn.MSELoss(reduction='none')
        loss_train = criterion(embs1[idx_train], embs2[idx_train]).sum(1).reshape(train_num, -1)
        loss_pos = criterion(embs1[idx_train], embs2[idx_train]).sum(1).reshape(train_num, -1).sum(1).mean()

        sample_nodes_set = self.sample_nodes # np.arange(0, self.node_num

        neg_left, neg_right, neg2_left, neg2_right = utils.negative_sample(sample_nodes_set, train,
                                                                           num_sample=self.num_sample)
        loss_neg_1 = -criterion(embs1[neg_left], embs2[neg_right]).sum(1).reshape(train_num, self.num_sample)
        loss_temp1 = torch.relu(loss_train + self.gamma + loss_neg_1)
        loss_neg_2 = -criterion(embs1[neg2_left], embs2[neg2_right]).sum(1).reshape(train_num, self.num_sample)
        loss_temp2 = torch.relu(loss_train + self.gamma + loss_neg_2)
        loss_train = (loss_temp1 + loss_temp2).mean() / 2
        print("loss_train:{:.2f} loss_pos:{:.2f} loss_neg:{:.2f} loss_test:{:.2f}".format(
                                                                           loss_train.item(),
                                                                           loss_pos,
                                                                           loss_neg_1.mean(),
                                                                           loss_temp1.mean()))

        return loss_train



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, feat_num, hidden_size, dropout, max_len=30):
        super(PositionalEncoding, self).__init__()
        # graph encoders which encodes a network snapshot to a vector representation
        self.feat_num = feat_num
        self.hidden_size = hidden_size
        self.d_model = d_model
        self.graphEncoder = GraphEncoder(feat_number=feat_num, hidden_size=hidden_size, emb_size=d_model)

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.Tensor(max_len, d_model)
        pe = nn.Parameter(pe)
        pe = nn.init.xavier_uniform_(pe)
        self.pe = pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x, features, adjs):
        alpha = 0.01
        beta = 0.1
        graph_embs = self.graphEncoder(features, adjs)
        x = x + alpha * self.pe[:x.size(0), :]
        x = x + beta * graph_embs
        return self.dropout(x)


class GraphEncoder(nn.Module):

    def __init__(self, feat_number, hidden_size, emb_size):
        super(GraphEncoder, self).__init__()
        self.feat_number = feat_number
        self.hidden_size = hidden_size
        self.emb_size = int(emb_size/2)
        self.gc_1 = GraphConvolution(self.feat_number, self.hidden_size)
        self.gc_2 = GraphConvolution(self.hidden_size, self.emb_size)

    def forward(self, features, adjs):
        graph_embs = F.relu(self.gc_1(features, adjs))
        graph_embs = self.gc_2(graph_embs, adjs)
        embs_mean = torch.mean(graph_embs, dim=1)
        embs_max = torch.max(graph_embs, dim=1)[0]
        graph_embs = torch.cat([embs_mean, embs_max], -1)
        graph_embs = graph_embs.unsqueeze(0).transpose(0, 1)
        return graph_embs


class DGAM_RNN(nn.Module):
    def __init__(self,
                 node_num,
                 hidden_size,
                 emb_size,
                 dropout,
                 num_layers=1,
                 gamma=3.0,
                 num_sample=5,
                 n_head=8,
                 dim_feedforward=512,
                 sample_nodes=None):
        super(DGAM_RNN, self).__init__()
        self.node_num = node_num
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.dropout = dropout
        self.num_layers = num_layers
        self.gamma = gamma
        self.n_head = n_head
        self.num_sample = num_sample
        self.dim_feedforward = dim_feedforward
        self.sample_nodes = sample_nodes
        self.weight_gcn_1 = nn.Parameter(torch.FloatTensor(self.node_num, self.hidden_size))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size,
                                                        nhead=self.n_head,
                                                        dim_feedforward=self.dim_feedforward,
                                                        dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.LSTM = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            batch_first=True,
                            bidirectional=False)
        self.gc_1 = GraphConvolution(self.node_num, self.hidden_size)
        self.gc_2 = GraphConvolution(self.hidden_size, self.emb_size)

    def forward(self, features, adjs, train):

        train_num = len(train)

        embs_l1 = F.relu(self.gc_1(features, adjs))

        embs_l1_1 = embs_l1[:, train_num:, :]
        embs_l2_2 = embs_l1[:, :train_num, :]

        embs_trans_1, (h_n_a, c_n_a) = self.LSTM(embs_l1_1)

        features = torch.cat((embs_trans_1, embs_l2_2), dim=1)
        features = F.dropout(features, self.dropout, training=self.training)

        self.embs = self.gc_2(features, adjs)

        return self.embs

    def calculate_loss(self, train=None, time_1=0, time_2=-1):

        train_num = len(train)
        embs1 = self.embs[time_1, :, :]
        embs2 = self.embs[time_2, :, :]
        idx_train = train[:, 0]

        criterion = nn.L1Loss(reduction='none')
        loss_train = criterion(embs1[idx_train], embs2[idx_train])
        loss_pos = criterion(embs1[idx_train], embs2[idx_train]).sum(1).reshape(train_num, -1)..mean()


        sample_nodes_set = self.sample_nodes # np.arange(0, self.node_num)

        neg_left, neg_right, neg2_left, neg2_right = utils.negative_sample(sample_nodes_set, train,
                                                                           num_sample=self.num_sample)
        loss_neg_1 = -criterion(embs1[neg_left], embs2[neg_right]).sum(1)
        loss_temp1 = torch.relu(loss_train + self.gamma + loss_neg_1)
        loss_neg_2 = -criterion(embs1[neg2_left], embs2[neg2_right]).sum(1).reshape(train_num, self.num_sample)
        loss_temp2 = torch.relu(loss_train + self.gamma + loss_neg_2)
        loss_train = (loss_temp1 + loss_temp2).mean() / 2
        print("loss_train:{} loss_pos:{} loss_neg:{} loss_test:{}".format(
                                                                           loss_train.item(),
                                                                           loss_pos,
                                                                           loss_neg_1.mean(),
                                                                           loss_temp1.mean()))

        return loss_train

    def evaluate(self, time_1=0, time_2=-1):
        start = self.embs[time_1, 500:, :]
        final = self.embs[time_2, 500:, :]
        # train = torch.randint(0, 500, (500, 2))
        criterion = nn.MSELoss()
        loss_train = criterion(start, final)
        # print(start.shape, final.shape)

        return loss_train






