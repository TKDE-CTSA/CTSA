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


def train(args):

    time_1 = args.time_1
    time_2 = args.time_2
    seed = 3
    dataset = args.dataset
    step_num = args.step_num  # adjs.shape[0]

    adjs = load_data.load_tn(dataset)
    node_num = adjs.shape[1]
    x = torch.eye(node_num).unsqueeze(0)
    x = x.repeat(step_num, 1, 1)


    node_num = adjs.shape[1]
    feat_num = x.shape[2]

    # create train and test

    train_num = int(node_num // 10 * seed)
    test_num = node_num - train_num
    train = np.array([np.arange(0, train_num), np.arange(0, train_num)]).transpose()
    sample_nodes = utils.get_test(np.arange(0, node_num), adjs[time_1], adjs[time_2])
    test = utils.get_test(np.arange(train_num, node_num), adjs[time_1], adjs[time_2])
    test = np.array([test, test]).transpose()

    features1_temp = torch.zeros(size=(test_num, feat_num)).type(torch.float)
    features2_temp = torch.zeros(size=(test_num, feat_num)).type(torch.float)
    features1 = torch.cat([x[time_1, 0:train_num, :], features1_temp], 0)
    features2 = torch.cat([x[time_2, 0:train_num, :], features2_temp], 0)

    adjs = load_data.preprocess_adjs(adjs)
    adjs = torch.from_numpy(adjs[0:step_num]).type(torch.float)
    x[time_1] = features1
    x[time_2] = features2

    # create model
    model = DGAM(node_num=node_num,
                 feat_num=feat_num,
                 hidden_size=args.hidden_size,
                 emb_size=args.emb_size,
                 dropout=args.dropout,
                 num_layers=args.num_layers,
                 gamma=args.gamma,
                 num_sample=args.num_sample,
                 n_head=args.n_head,
                 dim_feedforward=args.dim_feedforward,
                 sample_nodes=sample_nodes)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)

    # train model
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        model.forward(x, adjs, train)
        loss_train = model.calculate_loss(train, time_1, time_2)
        print("epoch:{} loss_train:{:2} ".format(epoch,
                                               loss_train.item(),
                                               ))
        loss_train.backward()
        optimizer.step()

    print("======" * 60)
    print("begin test")
    print("test_num:{}".format(test.shape[0]))
    embs = model.forward(x, adjs, train).data.numpy()
    embs1 = embs[time_1]
    embs2 = embs[time_2]
    utils.get_hits(embs1, embs2, test)
    # utils.save_embs(embs1, embs2, test)
    print("finish test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='model parameters')
    parser.add_argument("--hidden-size", type=int, default=512,
                        help="hidden size of gcn")
    parser.add_argument("--emb-size", type=int, default=256,
                        help="embedding size")
    parser.add_argument('--dropout', type=float, default=0.,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs to train.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers.')
    parser.add_argument('--time_2', type=int, default=2,
                        help='Time_2.')
    parser.add_argument('--time_1', type=int, default=2,
                        help='Time_1.')
    parser.add_argument('--step_num', type=int, default=2,
                        help='step_num.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of heads.')
    parser.add_argument('--gamma', type=float, default=3.0,
                        help='Number of layers.')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                        help='Number of layers.')
    parser.add_argument('--num_sample', type=int, default=5,
                        help='Number of negative samples.')

    args = parser.parse_args([])
    print(args)
    train(args)

