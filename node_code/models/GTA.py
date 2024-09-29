#%%
import pdb
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from node_code.models.GCN import GCN
from node_code.helpers.helpers import set_random_seed
from run_node_exps import args


#%%
class GradWhere(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, thrd, device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        rst = torch.where(input>thrd, torch.tensor(1.0, device=device, requires_grad=True),
                                      torch.tensor(0.0, device=device, requires_grad=True))
        return rst

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        
        """
        Return results number should corresponding with .forward inputs (besides ctx),
        for each input, return a corresponding backward grad
        """
        return grad_input, None, None

class GraphTrojanNet(nn.Module):
    # In the furture, we may use a GNN model to generate backdoor
    def __init__(self, device, nfeat, nout, layernum=1, dropout=0.00):
        super(GraphTrojanNet, self).__init__()
        set_random_seed(args.seed)  # 设置随机种子，确保权重初始化一致
        layers = []
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(layernum-1):
            layers.append(nn.Linear(nfeat, nfeat))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        self.layers = nn.Sequential(*layers).to(device)
        self.feat = nn.Linear(nfeat,nout*nfeat)
        self.edge = nn.Linear(nfeat, int(nout*(nout-1)/2))
        self.device = device
        self._initialize_weights()  # 初始化权重

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, thrd):

        """
        "input", "mask" and "thrd", should already in cuda before sent to this function.
        If using sparse format, corresponding tensor should already in sparse format before
        sent into this function
        """

        GW = GradWhere.apply
        self.layers = self.layers
        h = self.layers(input)

        feat = self.feat(h)
        edge_weight = self.edge(h)
        # feat = GW(feat, thrd, self.device)
        edge_weight = GW(edge_weight, thrd, self.device)

        return feat, edge_weight

class HomoLoss(nn.Module):
    def __init__(self,args,device):
        super(HomoLoss, self).__init__()
        self.args = args
        self.device = device
        
    def forward(self,trigger_edge_index,trigger_edge_weights,x,thrd):

        trigger_edge_index = trigger_edge_index[:,trigger_edge_weights>0.0]
        edge_sims = F.cosine_similarity(x[trigger_edge_index[0]],x[trigger_edge_index[1]])
        
        loss = torch.relu(thrd - edge_sims).mean()
        # print(edge_sims.min())
        return loss

#%%
import numpy as np


class Backdoor:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.weights = None
        set_random_seed(args.seed)  # 初始化时设置种子
        self.trigger_index = self.get_trigger_index(args.trigger_size)

    def get_trigger_index(self, trigger_size):
        edge_list = []
        edge_list.append([0, 0])
        for j in range(trigger_size):
            for k in range(j):
                edge_list.append([j, k])
        edge_index = torch.tensor(edge_list, device=self.device).long().T
        return edge_index

    def get_trojan_edge(self, start, idx_attach, trigger_size):
        edge_list = []

        for idx in idx_attach:
            edges = self.trigger_index.clone()
            edges[0, 0] = idx
            edges[1, 0] = start
            edges[:, 1:] = edges[:, 1:] + start

            edge_list.append(edges)
            start += trigger_size
        edge_index = torch.cat(edge_list, dim=1)
        # to undirected
        row = torch.cat([edge_index[0], edge_index[1]])
        col = torch.cat([edge_index[1], edge_index[0]])
        edge_index = torch.stack([row, col])
        return edge_index

    def inject_trigger(self, idx_attach, features, edge_index, edge_weight, device):
        self.trojan = self.trojan.to(device)
        idx_attach = idx_attach.to(device)
        features = features.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        self.trojan.eval()

        trojan_feat, trojan_weights = self.trojan(features[idx_attach],self.args.thrd)  # may revise the process of generate
        #print("trojan_feat before processing:", trojan_feat)  # My调试输出

        trojan_weights = torch.cat([torch.ones([len(idx_attach), 1], dtype=torch.float, device=device), trojan_weights],
                                   dim=1)
        trojan_weights = trojan_weights.flatten()
        trojan_feat = trojan_feat.view([-1, features.shape[1]])

        trojan_edge = self.get_trojan_edge(len(features), idx_attach, self.args.trigger_size).to(device)
        update_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])
        update_feat = torch.cat([features, trojan_feat])
        update_edge_index = torch.cat([edge_index, trojan_edge], dim=1)

        self.trojan = self.trojan.cpu()
        idx_attach = idx_attach.cpu()
        features = features.cpu()
        edge_index = edge_index.cpu()
        edge_weight = edge_weight.cpu()
        return update_feat, update_edge_index, update_edge_weights

    def fit(self, features, edge_index, edge_weight, labels, idx_train, idx_attach, idx_unlabeled):
        # 将所有张量移动到设备上
        features, edge_index, labels, idx_train, idx_attach, idx_unlabeled = (
            features.to(self.device), edge_index.to(self.device), labels.to(self.device),
            idx_train.to(self.device), idx_attach.to(self.device), idx_unlabeled.to(self.device)
        )

        args = self.args

        if edge_weight is None:
            edge_weight = torch.ones([edge_index.shape[1]], device=self.device, dtype=torch.float)
        self.idx_attach = idx_attach
        self.features = features
        self.edge_index = edge_index.to(self.device)
        self.edge_weights = edge_weight

        # 初始化一个 shadow 模型
        self.shadow_model = GCN(nfeat=features.shape[1],
                                nhid=self.args.hidden,
                                nclass=labels.max().item() + 1,
                                dropout=0.0, device=self.device).to(self.device)
        set_random_seed(self.args.seed)  # 再次设置种子以确保模型初始化的一致性

        # 初始化一个 trojanNet 用于生成触发器
        self.trojan = GraphTrojanNet(self.device, features.shape[1], args.trigger_size, layernum=2).to(self.device)
        set_random_seed(self.args.seed)  # 再次设置种子以确保模型初始化的一致性

        optimizer_shadow = torch.optim.Adam(self.shadow_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_trigger = torch.optim.Adam(self.trojan.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 更改被污染节点的标签为目标类别
        self.labels = labels.clone()
        self.labels[idx_attach] = args.target_class

        # 获取 Trojan 边，这些边包括目标-触发边和触发之间的边
        trojan_edge = self.get_trojan_edge(len(features), idx_attach, args.trigger_size).to(self.device)

        # 更新被污染图的边索引
        poison_edge_index = torch.cat([self.edge_index, trojan_edge], dim=1)

        loss_best = 1e8
        for i in range(args.trojan_epochs):
            self.trojan.train()
            for j in range(self.args.inner):
                set_random_seed(self.args.seed + i * self.args.inner + j)  # 为每个训练步骤设置种子
                optimizer_shadow.zero_grad()
                optimizer_trigger.zero_grad()
                trojan_feat, trojan_weights = self.trojan(features[idx_attach].to(self.device), args.thrd)
                trojan_weights = torch.cat(
                    [torch.ones([len(trojan_feat), 1], dtype=torch.float, device=self.device), trojan_weights], dim=1)
                trojan_weights = trojan_weights.flatten()
                trojan_feat = trojan_feat.view([-1, features.shape[1]])
                poison_edge_weights = torch.cat([edge_weight, trojan_weights, trojan_weights])  # 由于无向边重复 Trojan 权重
                poison_x = torch.cat([features.to(self.device), trojan_feat])

                output = self.shadow_model(poison_x, poison_edge_index, poison_edge_weights)
                loss_inner = F.nll_loss(output[torch.cat([idx_train, idx_attach])],
                                        self.labels[torch.cat([idx_train, idx_attach])])  # 添加自适应损失

                loss_inner.backward()
                optimizer_shadow.step()
                optimizer_trigger.step()

        self.trojan.eval()

    def get_poisoned(self):
        with torch.no_grad():

            poison_x, poison_edge_index, poison_edge_weights = self.inject_trigger(self.idx_attach, self.features,
                                                                                   self.edge_index, self.edge_weights,
                                                                                   self.device)
        poison_labels = self.labels
        poison_edge_index = poison_edge_index[:, poison_edge_weights > 0.0]
        poison_edge_weights = poison_edge_weights[poison_edge_weights > 0.0]
        return poison_x, poison_edge_index, poison_edge_weights, poison_labels


