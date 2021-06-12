import torch
import numpy as np
import scipy
import scipy.sparse as sp

def bdsmm(s_matrices, dense):
    outs = []
    for sparse in s_matrices:
        outs.append(torch.sparse.mm(sparse, dense))
    result = torch.stack(outs, 0)
    return result


def negative_sample(nodes, train_pos, num_sample=5):
    t = len(train_pos)
    k = num_sample
    L = np.ones((t, k)) * (train_pos[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (train_pos[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    neg2_left = np.random.choice(nodes, t * k)
    neg_right = np.random.choice(nodes, t * k)
    return neg_left, neg_right, neg2_left, neg2_right


def find_zero_row(adj):
    adj = adj.sum(1)
    index = np.where(adj == 0)
    return index


def find_zero_column(adj):
    adj = adj.sum(0)
    index = np.where(adj == 0)
    return index


def get_test(test, adj_1, adj_2):
    set_1 = set(find_zero_row(adj_1 + adj_1.T)[0]) # 已经对称化处理
    set_2 = set(find_zero_row(adj_2 + adj_2.T)[0])
    nodes = np.arange(0, adj_2.shape[0])
    # print(len(set_1))
    # print(len(set_2))
    print("all node to align: {}".format(len(nodes) - len(set_1 | set_2)))
    test = list(set(test) - set_1 - set_2)
    return test


def get_repeated_nodes(node_list, adj_1, adj_2):
    set_1 = set(find_zero_row(adj_1 + adj_1.T)[0])  # 对称化
    set_2 = set(find_zero_row(adj_2 + adj_2.T)[0])
    nodes = np.arange(0, adj_2.shape[0])
    # print(len(set_1))
    # print(len(set_2))
    common_nodes = list(set(node_list) - set_1 - set_2)
    union_nodes = list(set(node_list) - set_1.intersection(set_2))

    return len(common_nodes), len(union_nodes)


def read_edgelist(node_num, dataset):
    adj = np.zeros([node_num, node_num])
    file = open("data/{}.edgelist".format(dataset), 'r')
    lines = file.readlines()
    for line in lines:
        node1, node2 = line.split(' ')
        node1 = int(node1)
        node2 = int(node2)
        adj[node1][node2] = 1
        adj[node2][node1] = 1

    return adj


def write_edgelist(adj, dataset):

    file = open("data_b/{}.edgelist".format(dataset), 'w')
    index = np.nonzero(adj)
    edge_num = len(index[0])
    #     lines = []
    for i in range(edge_num):
        line = str(index[0][i]) + ' ' + str(index[1][i]) + '\n'
        file.write(line)
    #         print(line)
    #         lines.append(line)

    return 0


def write_edgelist_with_layer(adj, dataset, layer_id):

    file = open("data_m/{}.edgelist".format(dataset), 'w')
    index = np.nonzero(adj)
    edge_num = len(index[0])
    #     lines = []
    for i in range(edge_num):
        line = str(layer_id) + ' ' + str(index[0][i]) + ' ' + str(index[1][i]) + ' ' + str(1) + '\n'
        file.write(line)
    #         print(line)
    #         lines.append(line)

    return 0


def write_ids(ids, dataset):

    file = open("data_b/{}.ids".format(dataset), 'w')
    id_num = len(ids)
    #     lines = []
    for i in range(id_num):
        line = str(ids[i]) + '\n'
        file.write(line)
    #         print(line)
    #         lines.append(line)

    return 0


def create_data_na(adj_1, adj_2, seed=3, dataset=None):
    '''
    create data for network alignment baselines
    '''
    node_num = adj_1.shape[0]
    write_edgelist(adj_1, dataset+"1")
    write_edgelist(adj_2, dataset+"2")
    aligns = np.array([np.arange(0, node_num), np.arange(0, node_num)]).transpose()
    align_num = int(aligns.shape[0] // 10 * seed)
    to_align_num = node_num - align_num
    align_ids = np.arange(0, align_num)
    all_align_ids = get_test(np.arange(0, node_num), adj_1, adj_2)
    train_ids = get_test(np.arange(0, align_num), adj_1, adj_2)
    test_ids = get_test(np.arange(align_num, node_num), adj_1, adj_2)
    print("all_ids:{}, train_ids:{}, test_ids:{}".format(len(all_align_ids), len(train_ids), len(test_ids)))
    write_ids(train_ids, dataset+"_train_ids")
    write_ids(test_ids, dataset+"_test_ids")
    return 0


def create_data_na_all(adj_1, adj_2, adj_all, seed=3, dataset=None):
    '''
    create data for network alignment baselines
    '''
    node_num = adj_1.shape[0]
    write_edgelist(adj_all, dataset+"1")
    write_edgelist(adj_all, dataset+"2")
    aligns = np.array([np.arange(0, node_num), np.arange(0, node_num)]).transpose()
    align_num = int(aligns.shape[0] // 10 * seed)
    to_align_num = node_num - align_num
    align_ids = np.arange(0, align_num)
    all_align_ids = get_test(np.arange(0, node_num), adj_1, adj_2)
    # train_ids = get_test(np.arange(0, align_num), adj_1, adj_2)
    train_ids = np.arange(0, align_num)
    test_ids = get_test(np.arange(align_num, node_num), adj_1, adj_2)
    print("all_ids:{}, train_ids:{}, test_ids:{}".format(len(all_align_ids), len(train_ids), len(test_ids)))
    write_ids(train_ids, dataset+"_train_ids")
    write_ids(test_ids, dataset+"_test_ids")
    return 0


def create_aligns(node_num):
    aligns = np.array([np.arange(0, node_num), np.arange(0, node_num)]).transpose()
    return aligns


def wtrite_aligns(file_name, aligns):
    file = open(file_name, 'w')
    index = aligns.transpose()
    edge_num = len(index[0])
    for i in range(edge_num):
        line = str(index[0][i]) + '\t' + str(index[1][i]) + '\n'
        file.write(line)

    return 0


def wtrite_triples(file_name, adj, ids_dic):
    file = open(file_name, 'w')
    index = np.nonzero(adj)
    edge_num = len(index[0])
    for i in range(edge_num):
        relation = 0
        if index[0][i] == index[1][i]:
            relation = 1
        line = str(ids_dic[index[0][i]]) + '\t' + str(relation) + '\t' + str(ids_dic[index[1][i]]) + '\n'
        file.write(line)

    return 0


def create_dataset_kg(adj_1, adj_2, seed=3, dataset=None):
    file_dir = "data/kg/" + dataset + "/"
    node_num = adj_1.shape[0]
    adj_1 = adj_1 + np.eye(node_num)
    adj_2 = adj_2 + np.eye(node_num)
    aligns = np.array([np.arange(0, node_num), np.arange(0, node_num)]).transpose()
    triple_num = aligns.shape[0]
    align_num = int(aligns.shape[0] // 10 * seed)
    to_align_num = node_num - align_num
    # not_align_ids_1 = np.arange(0, to_align_num)
    # not_align_ids_2 = np.arange(to_align_num, to_align_num * 2)
    # align_ids = np.arange(to_align_num * 2, node_num + to_align_num)
    align_ids = np.arange(0, align_num)
    not_align_ids_1 = np.arange(align_num, align_num+to_align_num)
    not_align_ids_2 = np.arange(node_num, node_num+to_align_num)
    ref_ent_ids = np.concatenate(([not_align_ids_1], [not_align_ids_2]), axis=0).transpose()
    sup_ent_ids = np.concatenate(([align_ids], [align_ids]), axis=0).transpose()
    print("aligns:{} to_aligns:{}".format(align_num, to_align_num))
    wtrite_aligns(file_dir + "ref_ent_ids", ref_ent_ids)
    wtrite_aligns(file_dir + "sup_ent_ids", sup_ent_ids)
    ids_dic_1 = np.concatenate((align_ids, not_align_ids_1), 0)
    ids_dic_2 = np.concatenate((align_ids, not_align_ids_2), 0)
    wtrite_triples(file_dir + "triples_1", adj_1, ids_dic_1)
    wtrite_triples(file_dir + "triples_2", adj_2, ids_dic_2)
    test = get_test(np.arange(align_num, node_num), adj_1 - np.eye(node_num), adj_2-np.eye(node_num))
    test = np.array(test)
    test = np.array([test + align_num, test+node_num]).transpose()
    print("test_Num:{}".format(test.shape[0]))
    wtrite_aligns(file_dir + "ref_ent_ids", np.array(test))
    wtrite_aligns(file_dir + "sup_ent_ids", sup_ent_ids)
    return test


#     file_tri = open("data/{}.edgelist".format(dataset), 'r')


def get_hits(emb, emb2, test_pair, top_k=(1, 5, 10, 30, 50)):
    Lvec = np.array([emb[e1] for e1, e2 in test_pair])
    Rvec = np.array([emb2[e2] for e1, e2 in test_pair])
    print("test_pairs", len(test_pair))
    num = 0
    sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    # sim = scipy.spatial.distance.cdist(Lvec, Rvec, metric='euclidean')
    top_lr = [0] * len(top_k)
    rank_list_lr = []
    rank_list_rl = []
    for i in range(Lvec.shape[0]):
        rank = sim[i, :].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1
        rank_list_lr.append(rank_index+1)
    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank = sim[:, i].argsort()
        rank_index = np.where(rank == i)[0][0]
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1
        rank_list_rl.append(rank_index+1)
    rank_list_lr = np.array(rank_list_lr)
    rank_list_rl = np.array(rank_list_rl)
    print('For each left:')
    print("MR", rank_list_lr.mean())
    print("MRR", (1/rank_list_lr).mean())
    for i in range(len(top_lr)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_pair) * 100))
    print('For each right:')
    print("MR", rank_list_rl.mean())
    print("MRR", (1 / rank_list_rl).mean())
    for i in range(len(top_rl)):
        print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_pair) * 100))
    index = np.where(rank_list_lr<=10)
    save_embs(Lvec, Rvec, index[0])
    return rank_list_lr

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def load_adjs_and_attrs(dataset):
    """ initialize return tensors"""
    adjs = []
    attributes = []

    """ set the start and end of the period """
    start = 0
    end = 0

    if "DBLP" in dataset:
        start = 2005
        end = 2019

    for time in range(start, end):
        print("loading time {}".format(time))
        edge_path = "data/{}/{}.edge.{}".format(dataset, dataset, time)
        attri_path = "data/{}/{}.attr.{}".format(dataset, dataset, time)

        """ file_path """
        adj, attribute = load_AN(dataset, edge_path, attri_path)
        adjs.append(adj.todense())
        attributes.append(attribute.todense())

    adjs = np.array(adjs)
    attributes = np.array(attributes)
    attribute_overall = attributes.sum(axis=0)
    return adjs, attributes, attribute_overall


def load_AN(dataset, edge_path, attri_file):
    edge_file = open(edge_path, 'r')
    attri_file = open(attri_file, 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[0].split('\t')[1].strip())
    association_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{},association_num:{}". \
          format(dataset, node_num, edge_num, attribute_number, association_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    return adj, attribute


def save_embs(embs1, embs2, test_id):
    emb_source = embs1[test_id]
    emb_target = embs2[test_id]
    np.save("embs/emb_source", emb_source)
    np.save("embs/emb_target", emb_target)
    print("finish saving embeddings")
    return True

