import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from node_code.helpers.func_utils import accuracy
from copy import deepcopy
from torch_geometric.nn import GCNConv
import pdb

from run_node_exps import args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, layer=2, device=None,
                 layer_norm_first=True, use_ln=True):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"

        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(nfeat, nhid))
        self.lns = nn.ModuleList()
        self.lns.append(torch.nn.LayerNorm(nfeat))
        for _ in range(layer - 2):
            self.convs.append(GCNConv(nhid, nhid))
            self.lns.append(nn.LayerNorm(nhid))
        self.lns.append(nn.LayerNorm(nhid))
        self.gc2 = GCNConv(nhid, nclass)
        self.dropout = dropout
        self.lr = lr
        self.output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.weight_decay = weight_decay
        self.layer_norm_first = layer_norm_first
        self.use_ln = use_ln

        self._initialize_weights()  # 初始化权重

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, GCNConv):
                nn.init.kaiming_normal_(m.lin.weight)
                if m.lin.bias is not None:
                    nn.init.constant_(m.lin.bias, 0)

    def forward(self, x, edge_index, edge_weight=None):
        x.requires_grad_(True)
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        log_softmax_output = F.log_softmax(x, dim=1)
        return log_softmax_output

    def forward_energy(self, x, edge_index, edge_weight=None):
        x.requires_grad_(True)
        if self.layer_norm_first:
            x = self.lns[0](x)
        i = 0
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
            if self.use_ln:
                x = self.lns[i + 1](x)
            i += 1
            x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index, edge_weight)
        p = x.logsumexp(dim=1)
        return p

    def get_h(self, x, edge_index):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

    def fit(self, global_model, features, edge_index, edge_weight, aug_edge_index, aug_edge_weight, labels, idx_train,
            args, idx_val=None, train_iters=200, verbose=False):
        set_random_seed(args.seed)  # 在模型初始化时设置种子

        self.edge_index, self.edge_weight = edge_index, edge_weight  # Original graph
        self.aug_edge_index, self.aug_edge_weight = aug_edge_index, aug_edge_weight  # Augmented graph
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)

        if idx_val is None:
            self._train_without_val(self.labels, idx_train, train_iters, verbose)
        else:
            loss_train, loss_val, acc_train, acc_val = self._train_with_val(self, global_model, features, labels,
                                                                            idx_train, idx_val, edge_index, edge_weight,
                                                                            aug_edge_index, aug_edge_weight,
                                                                            train_iters, verbose, args)
        return loss_train, loss_val, acc_train, acc_val

    def _train_without_val(self, labels, idx_train, train_iters, verbose):
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for i in range(train_iters):
            set_random_seed(args.seed + i)  # 为每个训练步骤设置种子
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward(self.features, self.edge_index, self.edge_weight)
        self.output = output

    def _train_with_val(self, global_model, features, labels, idx_train, idx_val, edge_index, edge_weight,
                        aug_edge_index, aug_edge_weight, train_iters, args, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        best_loss_val = 100
        best_acc_val = -10

        for i in range(train_iters):
            set_random_seed(args.seed + i)  # 为每个训练步骤设置种子
            self.train()
            optimizer.zero_grad()
            output = self.forward(features, edge_index, edge_weight)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])

            total_loss = loss_train
            total_loss.backward()
            optimizer.step()

            self.eval()
            with torch.no_grad():
                output = self.forward(features, edge_index, edge_weight)
                loss_val = F.nll_loss(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])
                acc_train = accuracy(output[idx_train], labels[idx_train])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

        return loss_train.item(), loss_val.item(), acc_train, acc_val


    def adjust_bn_layers(self, features, edge_index, edge_weight, aug_edge_index, aug_edge_weight):
        bn_params = []
        num_nodes = features.size(0)
        for name, param in self.named_parameters():
            if 'lns' in name:  # LayerNorm层
                bn_params.append(param)
        optimizer = optim.Adam(bn_params, lr=self.lr, weight_decay=self.weight_decay)
        self.train()
        optimizer.zero_grad()
        p_data = self.forward_energy(features, edge_index, edge_weight)
        shuf_feats = features[:, torch.randperm(features.size(1))]  # shuffle features
        p_neigh = self.forward_energy(shuf_feats, aug_edge_index, aug_edge_weight)
        energy = p_data - p_neigh / p_data
        features.requires_grad_(True)
        energy_grad = torch.autograd.grad(energy.sum(), features, create_graph=True)[0]
        energy_grad_inner = torch.sum(energy_grad ** 2)
        energy_squared_sum = torch.sum(energy ** 2)
        neigh_loss = 1 / num_nodes * (energy_grad_inner + 1 / 2 * energy_squared_sum)
        neigh_loss.backward()
        optimizer.step()

    def test(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        with torch.no_grad():
            output = self.forward(features, edge_index, edge_weight)
            acc_test = accuracy(output[idx_test], labels[idx_test])

        return float(acc_test)

    def test_with_correct_nodes(self, features, edge_index, edge_weight, labels, idx_test):
        self.eval()
        output = self.forward(features, edge_index, edge_weight)
        correct_nids = (output.argmax(dim=1)[idx_test] == labels[idx_test]).nonzero().flatten()  # return a tensor
        acc_test = accuracy(output[idx_test], labels[idx_test])
        return acc_test, correct_nids
