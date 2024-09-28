import copy
import random
import numpy as np
import scipy as sp
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def aug_random_edge(input_adj, perturb_percent=0.2, drop_edge=True, add_edge=True, self_loop=True, seed=None):
    if seed is not None:
        set_random_seed(seed)

    aug_adj = copy.deepcopy(input_adj)
    nb_nodes = input_adj.shape[0]
    edge_index = (input_adj > 0).nonzero().t()
    edge_dict = {}
    for i in range(nb_nodes):
        edge_dict[i] = set()

    for edge in edge_index:
        i, j = edge[0], edge[1]
        i = i.item()
        j = j.item()
        edge_dict[i].add(j)
        edge_dict[j].add(i)

    if drop_edge:
        for i in range(nb_nodes):
            d = len(edge_dict[i])
            node_list = list(edge_dict[i])
            num_edge_to_drop = int(d * perturb_percent)

            sampled_nodes = random.sample(node_list, num_edge_to_drop)

            for j in sampled_nodes:
                aug_adj[i][j] = 0
                aug_adj[j][i] = 0

    node_list = [i for i in range(nb_nodes)]
    add_list = []
    for i in range(nb_nodes):
        d = len(edge_dict[i])
        num_edge_to_add = int(d * perturb_percent)
        sampled_nodes = random.sample(node_list, num_edge_to_add)
        for j in sampled_nodes:
            add_list.append((i, j))

    if add_edge:
        for i in add_list:
            aug_adj[i[0]][i[1]] = 1
            aug_adj[i[1]][i[0]] = 1

    if self_loop:
        for i in range(nb_nodes):
            aug_adj[i][i] = 1
            aug_adj[i][i] = 1

    return aug_adj


def _aug_random_edge(nb_nodes, edge_index, perturb_percent=0.2, drop_edge=True, add_edge=True, self_loop=True,
                     use_avg_deg=True, seed=None):
    if seed is not None:
        set_random_seed(seed)

    total_edges = edge_index.shape[1]
    avg_degree = int(total_edges / nb_nodes)

    edge_dict = {}
    for i in range(nb_nodes):
        edge_dict[i] = set()

    for edge in edge_index:
        i, j = edge[0], edge[1]
        i = i.item()
        j = j.item()
        edge_dict[i].add(j)
        edge_dict[j].add(i)

    if drop_edge:
        for i in range(nb_nodes):
            d = len(edge_dict[i])
            if use_avg_deg:
                num_edge_to_drop = avg_degree
            else:
                num_edge_to_drop = int(d * perturb_percent)

            node_list = list(edge_dict[i])
            num_edge_to_drop = min(num_edge_to_drop, d)
            sampled_nodes = random.sample(node_list, num_edge_to_drop)

            for j in sampled_nodes:
                edge_dict[i].discard(j)
                edge_dict[j].discard(i)

    node_list = [i for i in range(nb_nodes)]

    add_list = []
    for i in range(nb_nodes):
        if use_avg_deg:
            num_edge_to_add = avg_degree
        else:
            d = len(edge_dict[i])
            num_edge_to_add = int(d * perturb_percent)

        sampled_nodes = random.sample(node_list, num_edge_to_add)
        for j in sampled_nodes:
            add_list.append((i, j))

    if add_edge:
        for edge in add_list:
            u = edge[0]
            v = edge[1]
            edge_dict[u].add(v)
            edge_dict[v].add(u)

    if self_loop:
        for i in range(nb_nodes):
            edge_dict[i].add(i)

    updated_edges = set()
    for i in range(nb_nodes):
        for j in edge_dict[i]:
            updated_edges.add((i, j))
            updated_edges.add((j, i))

    row = []
    col = []
    for edge in updated_edges:
        u = edge[0]
        v = edge[1]
        row.append(u)
        col.append(v)

    aug_edge_index = [row, col]
    aug_edge_index = torch.tensor(aug_edge_index)

    return aug_edge_index
def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features = features.squeeze()
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
    # return features, sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

