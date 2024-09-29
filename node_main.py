import torch
import numpy as np
import torch_geometric.transforms as T
import wandb
from torch_geometric.utils import scatter
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
import node_code.helpers.selection_utils  as hs
from node_code.helpers.func_utils import subgraph,get_split
from torch_geometric.utils import to_undirected
from node_code.helpers.split_graph_utils import split_Random, split_Louvain, split_Metis
from node_code.models.construct import model_construct
from node_code.helpers.func_utils import prune_unrelated_edge,prune_unrelated_edge_isolated
from node_code.data.datasets import  ogba_data,Amazon_data,Coauthor_data
from node_code.helpers.helpers import _aug_random_edge, set_random_seed
from node_code.helpers.select_models_by_energy import select_models_based_on_energy
from node_code.aggregators.aggregation import fed_EnergyBelief


def main(args, logger,round):

    set_random_seed(args.seed)

    Coauthor_list = ["Cs","Physics"]
    Amazon_list = ["photo"]
    ##### DATA PREPARATION #####
    if (args.dataset == 'Cora'  or args.dataset == 'Pubmed'):
        dataset = Planetoid(root='./data/', \
                            name=args.dataset, \
                            transform=T.LargestConnectedComponents())
    elif (args.dataset in Coauthor_list):
        dataset = Coauthor(root='./data/',name =args.dataset,  \
                          transform=T.NormalizeFeatures())
        print('datasets', dataset[0])
    elif (args.dataset in Amazon_list):
        dataset = Amazon(root='./data/',name =args.dataset,  \
                          transform=T.NormalizeFeatures())

    print(f'Dataset: {dataset}:')
    print('======================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    ogbn_data_list = ["ogbn-arxiv",'ogbn-products','ogbn-proteins']
    if args.dataset in ogbn_data_list:
        data = ogba_data(dataset)
    elif args.dataset in Amazon_list:
        data = Amazon_data(dataset)
        data.y = data.y.to(dtype=torch.long)
    elif args.dataset in Coauthor_list:
        data = Coauthor_data(dataset)
    else:
        data = dataset[0]  # Get the graph object.
    if args.dataset == 'ogbn-proteins':
        row, col = data.edge_index
        data.x = scatter(data.edge_attr, col, dim_size=data.num_nodes, reduce='sum')
        _, f_dim = data.x.size()
        print(f'ogbn-proteins Number of features: {f_dim}')
        print("data.y = data.y.to(torch.float)", data.y.shape)
    if args.dataset == 'Reddit':
        data.y = data.y.long()
    args.avg_degree = data.num_edges / data.num_nodes
    nclass = int(data.y.max() + 1)
    print("class", int(data.y.max() + 1))
    print('==============================================================')
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print('======================Start Splitting the Data========================================')

    if args.is_iid == "iid":
        client_data = split_Random(args, data)
    elif args.is_iid == "non-iid-louvain":
        client_data = split_Louvain(args, data)
    elif args.is_iid == "non-iid-Metis":
        client_data = split_Metis(args, data)
    else:
        raise NameError

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device_id)

    for i in range(args.num_workers):
        print(len(client_data[i]))

    print('======================Start Preparing the Data========================================')
    for i in range(args.num_workers):
        print("Client:{}".format(i))
        print(client_data[i])
        print(f'Number of nodes: {client_data[i].num_nodes}')
        print(f'Number of edges: {client_data[i].num_edges}')
        print(f'Number of train: {client_data[i].train_mask.sum()}')
        print(f'Number of val: {client_data[i].val_mask.sum()}')
        print(f'Number of test: {client_data[i].test_mask.sum()}')

    client_train_edge_index = []
    client_edge_mask = []
    client_mask_edge_index = []
    client_unlabeled_idx = []
    client_idx_train, client_idx_val, client_idx_clean_test, client_idx_atk = [], [], [], []
    for k in range(len(client_data)):
        data, idx_train, idx_val, idx_clean_test, idx_atk = get_split(args,client_data[k],device)
        client_idx_train.append(idx_train)
        client_idx_val.append(idx_val)
        client_idx_clean_test.append(idx_clean_test)
        client_idx_atk.append(idx_atk)

        edge_weight = torch.ones([data.edge_index.shape[1]], device=device, dtype=torch.float)
        data.edge_weight = edge_weight
        data.edge_index = to_undirected(data.edge_index)
        train_edge_index,_, edge_mask = subgraph(torch.bitwise_not(data.test_mask),data.edge_index,relabel_nodes=False)
        mask_edge_index = data.edge_index[:,torch.bitwise_not(edge_mask)]
        client_data[k] = data
        client_train_edge_index.append(train_edge_index)
        client_edge_mask.append(edge_mask)
        client_mask_edge_index.append(mask_edge_index)
        unlabeled_idx = (torch.bitwise_not(data.test_mask)&torch.bitwise_not(data.train_mask)).nonzero().flatten()
        client_unlabeled_idx.append(unlabeled_idx)
    for i in range(args.num_workers):
        print("Client:{}".format(i))
        print(client_data[i])
            # Gather some statistics about the graph.
        print(f'Number of nodes: {client_data[i].num_nodes}')
        print(f'Number of edges: {client_data[i].num_edges}')
        print(f'Number of train: {client_data[i].train_mask.sum()}')
        print(f'Number of val: {client_data[i].val_mask.sum()}')
        print(f'Number of test: {client_data[i].test_mask.sum()}')

    augmented_clean_edge_indices = []
    augmented_clean_edge_weights = []
    for k in range(len(client_data)):
        set_random_seed(args.seed + k)

        clean_edge_index = client_train_edge_index[k].to(device)
        num_nodes = client_data[k].x.size(0)
        aug_clean_edge_index = _aug_random_edge(num_nodes, clean_edge_index, perturb_percent=0.2,
                                                drop_edge=True, add_edge=True, self_loop=True, use_avg_deg=True,seed=args.seed)
        aug_clean_edge_index = aug_clean_edge_index.to(device)
        aug_clean_edge_weights = torch.ones(aug_clean_edge_index.size(1), dtype=torch.float, device=device)
        augmented_clean_edge_indices.append(aug_clean_edge_index)
        augmented_clean_edge_weights.append(aug_clean_edge_weights)
        print("Client:{} Number of training nodes:{}".format(k,len(idx_train)))
        print("Client:{} Number of validation nodes:{}".format(k,len(idx_val)))
        print("Client:{} Number of testing nodes:{}".format(k,len(idx_clean_test)))
        print("Client:{} Number of unlabeled nodes:{}".format(k,len(unlabeled_idx)))
    print('======================Start Preparing the Backdoor Attack========================================')
    # prepare for malicious attacker
    Backdoor_model_list = []
    heuristic_trigger_list = ["renyi","ws", "ba"]
    for i in range(args.num_mali):
        if args.trigger_type== "gta":
           from node_code.models.GTA import Backdoor
           Backdoor_model = Backdoor(args, device)
        elif args.trigger_type == "ugba":
            from node_code.models.backdoor import Backdoor
            Backdoor_model = Backdoor(args, device)
        elif args.trigger_type in heuristic_trigger_list:
            from node_code.models.Heuristic import Backdoor
            Backdoor_model = Backdoor(args, device)
        else:
            raise NameError
        Backdoor_model_list.append(Backdoor_model)

    print('======================Start Preparing the Trigger Posistion========================================')
    client_idx_attach = []
    for i in range(args.num_workers):

        size =  int((len(client_unlabeled_idx[i]))*args.poisoning_intensity)
        if (args.trigger_position == 'random'):
            idx_attach = hs.obtain_attach_nodes(args, client_unlabeled_idx[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'learn_cluster'):
            idx_attach = hs.cluster_distance_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                       client_train_edge_index[i], size, device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'learn_cluster_degree'):
            idx_attach = hs.cluster_degree_selection(args, client_data[i], client_idx_train[i], client_idx_val[i], client_idx_clean_test[i], client_unlabeled_idx[i],
                                                       client_train_edge_index[i], size, device)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'degree'):
            idx_attach = hs.obtain_attach_nodes_degree(args, client_unlabeled_idx[i],client_data[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        elif (args.trigger_position == 'cluster'):
            idx_attach = hs.obtain_attach_nodes_cluster(args, client_unlabeled_idx[i],client_data[i], size)
            idx_attach = torch.LongTensor(idx_attach).to(device)
        else:
            raise NameError
        client_idx_attach.append(idx_attach)

    print('======================Start Preparing the Posioned Datasets========================================')
    # construct the triggers
    client_poison_x, client_poison_edge_index, client_poison_edge_weights, client_poison_labels = [], [], [], []
    for i in range(args.num_mali):
        set_random_seed(args.seed + i)
        backdoor_model = Backdoor_model_list[i]
        backdoor_model.fit(client_data[i].x,client_train_edge_index[i], None, client_data[i].y, client_idx_train[i], client_idx_attach[i], client_unlabeled_idx[i])
        poison_x, poison_edge_index, poison_edge_weights, poison_labels = backdoor_model.get_poisoned()
        client_poison_x.append(poison_x)
        client_poison_edge_index.append(poison_edge_index)
        client_poison_edge_weights.append(poison_edge_weights)
        client_poison_labels.append(poison_labels)

    # data level defense
    client_bkd_tn_nodes = []
    for i in range(args.num_mali):
        set_random_seed(args.seed + i)

        if (args.defense_mode == 'prune'):
            poison_edge_index, poison_edge_weights = prune_unrelated_edge(args, client_poison_edge_index[i], client_poison_edge_weights[i],
                                                                          client_poison_x[i], device, large_graph=False)

            bkd_tn_nodes = torch.cat([client_idx_train[i], client_idx_attach[i]]).to(device)
        elif (args.defense_mode == 'isolate'):
            poison_edge_index, poison_edge_weights, rel_nodes = prune_unrelated_edge_isolated(args, client_poison_edge_index[i],
                                                                                              client_poison_edge_weights[i], client_poison_x[i],
                                                                                              device, large_graph=False)
            bkd_tn_nodes = torch.cat([client_idx_train[i], client_idx_attach[i]]).tolist()
            bkd_tn_nodes = torch.LongTensor(list(set(bkd_tn_nodes) - set(rel_nodes))).to(device)
        else:
            poison_edge_weights = client_poison_edge_weights[i]
            poison_edge_index = client_poison_edge_index[i]
            bkd_tn_nodes = torch.cat([client_idx_train[i].to(device), client_idx_attach[i].to(device)])
        print("precent of left attach nodes: {:.3f}" \
              .format(len(set(bkd_tn_nodes.tolist()) & set(idx_attach.tolist())) / len(idx_attach)))

        client_poison_edge_index[i] = poison_edge_index
        client_poison_edge_weights[i] = poison_edge_weights
        client_bkd_tn_nodes.append(bkd_tn_nodes)

    print('======================Start Preparing the Aumented Posioned Datasets========================================')
    augmented_poison_edge_indices = []
    augmented_poison_edge_weights = []

    for i in range(args.num_mali):
        set_random_seed(args.seed + i)

        poison_edge_index = client_poison_edge_index[i].to(device)
        num_nodes = client_poison_x[i].size(0)

        aug_poison_edge_index = _aug_random_edge(num_nodes, poison_edge_index, perturb_percent=0.2,
                                                 drop_edge=True, add_edge=True, self_loop=True, use_avg_deg=True,seed=args.seed)
        aug_poison_edge_index = aug_poison_edge_index.to(device)
        aug_poison_edge_weights = torch.ones(aug_poison_edge_index.size(1), dtype=torch.float, device=device)

        augmented_poison_edge_indices.append(aug_poison_edge_index)
        augmented_poison_edge_weights.append(aug_poison_edge_weights)

    optimizer_list = []
    print('======================Start Preparing the Models========================================')
    model_list = []
    for i in range(args.num_workers):

        test_model = model_construct(args, args.model, data, device,nclass).to(device)
        model_list.append(test_model)

    global_model = model_construct(args, args.model, data, device,nclass).to(device)
    rs = [i for i in range(args.num_mali)]
    print("rs",rs)
    args.epoch_backdoor = int(args.epoch_backdoor * args.epochs)
    print(f'args.epochs={args.epochs}')
    print('======================Start EnergyGCN Training Model========================================')
    for epoch in range(args.epochs):

        client_induct_edge_index = []
        client_induct_edge_weights = []
        worker_results = {f"client_{i}": {"train_loss": None, "train_acc": None, "val_loss": None, "val_acc": None} for i in range(args.num_workers)}

        for param_tensor in global_model.state_dict():
            global_para = global_model.state_dict()[param_tensor]
            for local_model in model_list:
                local_model.state_dict()[param_tensor].copy_(global_para)

        if epoch >=1:
            client_energies = []
            is_malicious = []
            for client_id, model in enumerate(model_list):
                if client_id in rs:
                    data_x = client_poison_x[client_id].to(device)
                    data_edge_index = client_poison_edge_index[client_id].to(device)
                    data_edge_weight = client_poison_edge_weights[client_id].to(device) if client_poison_edge_weights[
                                                                                               client_id] is not None else None
                else:
                    data_x = client_data[client_id].x.to(device)
                    data_edge_index = client_data[client_id].edge_index.to(device)
                    data_edge_weight = client_data[client_id].edge_weight.to(device) if 'edge_weight' in client_data[
                        client_id] else None
                with torch.no_grad():
                    energies = model_list[client_id].forward_energy(data_x, data_edge_index,
                                                                    data_edge_weight).detach().cpu().numpy()
                    client_energies.append(energies)

                is_malicious.append(client_id in rs)

        if epoch >= args.epoch_backdoor:
            for j in range(args.num_workers):

                if j in rs:
                    loss_train, loss_val, acc_train, acc_val = model_list[j]._train_with_val(
                        global_model,
                        client_poison_x[j].to(device),
                        client_poison_labels[j].to(device),
                        client_bkd_tn_nodes[j].to(device),
                        client_idx_val[j].to(device),
                        client_poison_edge_index[j].to(device),
                        client_poison_edge_weights[j].to(device),
                        augmented_poison_edge_indices[j].to(device),
                        augmented_poison_edge_weights[j].to(device),
                        args.inner_epochs,
                        args,
                        verbose=False,
                    )
                    print(f"Malicious client: {j} ,Acc train: {acc_train:.4f}, Acc val: {acc_val:.4f}")

                    induct_edge_index = torch.cat([client_poison_edge_index[j].to(device), client_mask_edge_index[j].to(device)], dim=1)
                    induct_edge_weights = torch.cat([client_poison_edge_weights[j], torch.ones([client_mask_edge_index[j].shape[1]], dtype=torch.float, device=device)])

                    if args.is_energy:
                        for k in range(args.energy_epochs):
                            set_random_seed(args.seed + epoch + j + k)
                            model_list[j].adjust_bn_layers(
                                client_poison_x[j].to(device),
                                client_poison_edge_index[j].to(device),
                                client_poison_edge_weights[j].to(device),
                                augmented_poison_edge_indices[j].to(device),
                                augmented_poison_edge_weights[j].to(device)
                            )
                        _, _, acc_train, acc_val = model_list[j]._train_with_val(
                            global_model,
                            client_poison_x[j].to(device),
                            client_poison_labels[j].to(device),
                            client_bkd_tn_nodes[j].to(device),
                            client_idx_val[j].to(device),
                            client_poison_edge_index[j].to(device),
                            client_poison_edge_weights[j].to(device),
                            augmented_poison_edge_indices[j].to(device),
                            augmented_poison_edge_weights[j].to(device),
                            1,
                            args,
                            verbose=False,
                        )
                        print(f"Energy: Malicious client: {j} ,Acc train: {acc_train:.4f}, Acc val: {acc_val:.4f}")

                else:
                    train_edge_weights = torch.ones([client_train_edge_index[j].shape[1]]).to(device)
                    loss_train, loss_val, acc_train, acc_val = model_list[j]._train_with_val(
                        global_model,
                        client_data[j].x.to(device),
                        client_data[j].y.to(device),
                        client_idx_train[j].to(device),
                        client_idx_val[j].to(device),
                        client_train_edge_index[j].to(device),
                        train_edge_weights.to(device),
                        augmented_clean_edge_indices[j].to(device),
                        augmented_clean_edge_weights[j].to(device),
                        args.inner_epochs,
                        args,
                        verbose=False,
                    )
                    print(f"Clean client: {j} ,Acc train: {acc_train:.4f}, Acc val: {acc_val:.4f}")

                    if args.is_energy:
                        for k in range(args.energy_epochs):
                            set_random_seed(args.seed + epoch + j + k)
                            model_list[j].adjust_bn_layers(
                                client_data[j].x.to(device),
                                client_train_edge_index[j].to(device),
                                train_edge_weights.to(device),
                                augmented_clean_edge_indices[j].to(device),
                                augmented_clean_edge_weights[j].to(device)
                            )
                        _, _, acc_train, acc_val = model_list[j]._train_with_val(
                            global_model,
                            client_data[j].x.to(device),
                            client_data[j].y.to(device),
                            client_idx_train[j].to(device),
                            client_idx_val[j].to(device),
                            client_train_edge_index[j].to(device),
                            train_edge_weights.to(device),
                            augmented_clean_edge_indices[j].to(device),
                            augmented_clean_edge_weights[j].to(device),
                            1,
                            args,
                            verbose=False,
                        )
                        print(f"Energy: Clean client: {j} ,Acc train: {acc_train:.4f}, Acc val: {acc_val:.4f}")

                wandb.log({
                    f"client_{j}/train_loss": loss_train,
                    f"client_{j}/train_acc": acc_train,
                    f"client_{j}/val_loss": loss_val,
                    f"client_{j}/val_acc": acc_val,
                })

                worker_results[f"client_{j}"] = {
                    "train_loss": loss_train,
                    "train_acc": acc_train,
                    "val_loss": loss_val,
                    "val_acc": acc_val
                }

                client_induct_edge_index.append(induct_edge_index)
                client_induct_edge_weights.append(induct_edge_weights)

        if args.is_energy:

            print('==========================calculate the energy =========================')
            client_energies = []
            is_malicious = []

            for client_id, model in enumerate(model_list):
                if client_id in rs:
                    data_x = client_poison_x[client_id].to(device)
                    data_edge_index = client_poison_edge_index[client_id].to(device)
                    data_edge_weight = client_poison_edge_weights[client_id].to(device) if client_poison_edge_weights[
                                                                                               client_id] is not None else None
                else:
                    data_x = client_data[client_id].x.to(device)
                    data_edge_index = client_data[client_id].edge_index.to(device)
                    data_edge_weight = client_data[client_id].edge_weight.to(device) if 'edge_weight' in client_data[
                        client_id] else None

                with torch.no_grad():
                    energies = model_list[client_id].forward_energy(data_x, data_edge_index,data_edge_weight).detach().cpu().numpy()
                    client_energies.append(energies)
                is_malicious.append(client_id in rs)

        selected_models_index = select_models_based_on_energy(client_energies)
        selected_models = [model_list[i] for i in selected_models_index]
        selected_energies = [client_energies[i] for i in selected_models_index]
        print("current epoch:", epoch)
        print("selected id", selected_models_index)

        if args.agg_method == "EnergyBelief":
            global_model = fed_EnergyBelief(global_model, selected_models, selected_energies, args)
        else:
            raise NameError

    overall_performance = []
    overall_malicious_train_attach_rate = []
    overall_malicious_train_flip_asr = []

    for i in range(args.num_workers):
        client_data[i].y = client_data[i].y.to(device)
        induct_x, induct_edge_index, induct_edge_weights = client_data[i].x, client_data[i].edge_index, client_data[
            i].edge_weight
        accuracy = global_model.test(induct_x.to(device), induct_edge_index.to(device), induct_edge_weights.to(device),
                                     client_data[i].y.to(device), client_idx_clean_test[i].to(device))
        print("Client: {}, Accuracy: {:.4f}".format(i, accuracy))
        overall_performance.append(accuracy)

        if i in rs:
            idx_atk = client_idx_atk[i]
            induct_x, induct_edge_index, induct_edge_weights = Backdoor_model_list[i].inject_trigger(idx_atk,
                                                                                                     client_data[i].x,
                                                                                                     client_data[
                                                                                                         i].edge_index,
                                                                                                     client_data[
                                                                                                         i].edge_weight,
                                                                                                     device)

            output = global_model(induct_x, induct_edge_index, induct_edge_weights)
            train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
            print("ASR: {:.4f}".format(train_attach_rate))
            overall_malicious_train_attach_rate.append(train_attach_rate.cpu().numpy())

            flip_y = client_data[i].y[idx_atk].to(device)
            flip_idx_atk = idx_atk[(flip_y != args.target_class).nonzero().flatten()]
            flip_asr = (output.argmax(dim=1)[flip_idx_atk] == args.target_class).float().mean()
            print("Flip ASR: {:.4f}/{} nodes".format(flip_asr, flip_idx_atk.shape[0]))
            overall_malicious_train_flip_asr.append(flip_asr.cpu().numpy())

    average_performance = np.mean(overall_performance)
    average_asr = np.mean(overall_malicious_train_attach_rate)
    average_flip_asr = np.mean(overall_malicious_train_flip_asr)

    R = 1 - average_asr

    V =0.5* average_performance + 0.5*R

    print("args.aggregation_method: {}".format(args.agg_method))
    print("Average Performance (A): {:.4f}".format(average_performance))
    print("Backdoor Failure Rate (R): {:.4f}".format(R))
    print("Heterogeneity and Robustness Trade-off (V): {:.4f}".format(V))

    transfer_attack_success_rate_list = []
    if args.num_workers - args.num_mali <= 0:
        average_transfer_attack_success_rate = -10000.0
    else:
        for i in range(args.num_mali):
            for j in range(args.num_workers - args.num_mali):
                idx_atk = client_idx_atk[i]
                induct_x, induct_edge_index, induct_edge_weights = Backdoor_model_list[i].inject_trigger(
                    client_idx_atk[i], client_poison_x[i], client_induct_edge_index[i], client_induct_edge_weights[i],
                    device)

                output = model_list[args.num_mali + j](induct_x, induct_edge_index, induct_edge_weights)
                train_attach_rate = (output.argmax(dim=1)[idx_atk] == args.target_class).float().mean()
                print('Clean client %d with trigger %d: %.3f' % (args.num_mali + j, i, train_attach_rate))
                transfer_attack_success_rate_list.append(train_attach_rate.cpu().numpy())
        average_transfer_attack_success_rate = np.mean(np.array(transfer_attack_success_rate_list))

    print("Malicious client: {}".format(rs))
    print("Average ASR: {:.4f}".format(average_asr))
    print("Flip ASR: {:.4f}".format(average_flip_asr))
    print("Transfer ASR: {:.4f}".format(average_transfer_attack_success_rate))
    print("Average Performance on clean test set: {:.4f}".format(average_performance))

    average_overall_performance = average_performance
    average_ASR = average_asr
    average_Flip_ASR = average_flip_asr

    # Ensure to return four values
    return average_overall_performance, average_ASR, average_Flip_ASR, average_transfer_attack_success_rate,average_performance,R,V


if __name__ == '__main__':
    main()