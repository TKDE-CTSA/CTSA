import scipy.sparse as sp
import numpy as np
import pickle
import networkx as nx
import torch as th
from dgl import DGLGraph
from sklearn.model_selection import ShuffleSplit

import os
import re

import utils


def view_network(adjs):
    for index, adj in enumerate(adjs):
        print(index, np.array(np.array(adj).nonzero()).shape)

    temp = np.full(adjs[0].shape, 0)
    out = []
    for i in range(len(adjs)):
        temp += adjs[i].astype(int)
    for i in range(len(adjs)):
        out.append((np.array(np.where(temp==i)).shape))
    print(out)


def load_tn(dataset):
    print("="*20)
    print("load dataset:{}".format(dataset))
    if dataset in ["email", "college"]:
        file = dataset + ".edge"
        with open(file, 'rb') as f:
            adjs = pickle.load(f)

    elif dataset in ['email_20']:
        file = dataset + ".edge"
        with open(file, 'rb') as f:
            adjs = pickle.load(f)
    else:
        adjs = None
    print("load dataset:{} finished".format(dataset))
    print("time steps:{}".format(adjs.shape[0]))
    print("=" * 20)
    return adjs


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]) + adj.T)
    return adj_normalized


def preprocess_adjs(adjs):
    adjs_normalized = []
    for adj in adjs:
        adjs_normalized.append(preprocess_adj(adj))
    return np.array(adjs_normalized)






