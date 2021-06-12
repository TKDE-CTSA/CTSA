import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import *
import os
import load_data
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import utils
import sys
import argparse
import time
from sklearn.utils import shuffle
import pickle
import load_data

# datasets email email_20 wiki college

def train(args):
    adjs = load_data.load_tn("email_20")
    node_num = adjs.shape[2]
    seed = 3
    train_num = int(node_num // 10 * seed)
    test_num = node_num - train_num
    num_sample = 5
    time_1 = 0
    time_2 = 19

    eye_train = torch.eye(train_num, node_num)
    features1_temp = torch.zeros(size=(test_num, node_num)).type(torch.float)
    features2_temp = torch.zeros(size=(test_num, node_num)).type(torch.float)
    features1 = torch.cat([eye_train, features1_temp], 0)
    features2 = torch.cat([eye_train, features2_temp], 0)
    # 设置特征的数量， 这里不一定是节点的数量
    feature_num = node_num
    # features1 = torch.nn.Parameter(torch.randn(node_num, feature_num), requires_grad=True)
    # features2 = torch.nn.Parameter(torch.randn(node_num, feature_num), requires_grad=True)
    print("==" * 20)
    print("all node align:")
    all_align_nodes = utils.get_test(np.arange(0, node_num), adj_1=adjs[time_1], adj_2=adjs[time_2])
    test_align_nodes = utils.get_test(np.arange(train_num, node_num), adj_1=adjs[time_1], adj_2=adjs[time_2])
    print(len(utils.get_test(np.arange(0, node_num), adj_1=adjs[time_1], adj_2=adjs[time_2])))
    print("==" * 20)
    print("begin training")
    test_nodes = utils.get_test(np.arange(train_num, node_num), adjs[time_1], adjs[time_2])
    # test = utils.get_test(np.arange(0, node_num), adjs[time_1], adjs[time_2])
    test_pairs = np.array([test_nodes, test_nodes]).transpose()
    adjs = load_data.preprocess_adjs(adjs)
    adj1 = torch.from_numpy(adjs[time_1]).type(torch.float)
    adj2 = torch.from_numpy(adjs[time_2]).type(torch.float)
    gcn1 = GCN(feature_num, args.hidden, args.emb_size, args.dropout)
    gcn2 = GCN(feature_num, args.hidden, args.emb_size, args.dropout)
    optimizer = optim.Adam([{'params': gcn1.parameters()},
                            {'params': gcn2.parameters()},
                            {'params': features1},
                            {'params': features2}],
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        gcn1.train()


        optimizer.zero_grad()
        embs1 = gcn1.forward(features1, adj1)
        embs2 = gcn1.forward(features2, adj2)
        idx_train = torch.arange(0, train_num)
        idx_test = np.arange(train_num, node_num)
        # np.random.shuffle(idx_test)

        idx_test = torch.from_numpy(idx_test).type(torch.long)
        #print(idx_test)
        criterion = nn.L1Loss(reduction='none')

        loss_train = criterion(embs1[idx_train], embs2[idx_train]).sum(1).reshape(train_num, -1)
        loss_pos = criterion(embs1[idx_train], embs2[idx_train]).sum(1).reshape(train_num, -1).sum(1).mean()
        train = np.array([np.arange(0, train_num), np.arange(0, train_num)]).transpose()

        #neg sample
        gamma = 3.0
        sample_nodes = utils.get_test(np.arange(0, node_num), adjs[time_1], adjs[time_2])
        neg_left, neg_right, neg2_left, neg2_right = utils.negative_sample(sample_nodes, train, num_sample=num_sample)
        #print(criterion(embs1[neg_left], embs2[neg_right]).sum(1).shape)
        loss_neg_1 = -criterion(embs1[neg_left], embs2[neg_right]).sum(1).reshape(train_num, num_sample)
        loss_temp1 = torch.relu(loss_train + gamma + loss_neg_1)
        loss_neg_2 = -criterion(embs1[neg2_left], embs2[neg2_right]).sum(1).reshape(train_num, num_sample)
        loss_temp2 = torch.relu(loss_train + gamma + loss_neg_2)
        loss_train = (loss_temp1 + loss_temp2).mean() / 2

        #test
        loss_test = criterion(embs1[idx_test], embs2[idx_test]).sum(1).mean()
        print("epoch:{} loss_train:{} loss_pos:{} loss_neg:{} loss_test:{}".format(epoch,
                                                           loss_train.item(),
                                                           loss_pos,
                                                           loss_neg_1.mean(),
                                                           loss_temp1.mean()))
        loss_train.backward()
        optimizer.step()
    print("finish training")
    print("==="*10)
    print("begin test")

    # test = np.array([np.arange(train_num, node_num), np.arange(train_num, node_num)]).transpose()
    embs1 = gcn1.forward(features1, adj1).data.numpy()
    embs2 = gcn1.forward(features2, adj2).data.numpy()
    print("test_num:{}".format(test_pairs.shape[0]))
    utils.get_hits(embs1, embs2, test_pairs)
    print("finish test")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='model parameters')
    parser.add_argument("--hidden", type=int, default=512,
                        help="hidden size of gcn")
    parser.add_argument("--emb-size", type=int, default=256,
                        help="embedding size")
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epochs.')
    args = parser.parse_args([])
    print(args)
    train(args)
