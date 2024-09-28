import numpy as np
import torch
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import degree
from torch_sparse import SparseTensor, matmul

def compute_similarity_matrix(energies):
    """Calculate the similarity matrix between energies"""
    return cosine_similarity(energies)

def build_edge_index(similarity_matrix, threshold=0.85):
    """Construct edge indices based on the similarity matrix and threshold"""
    edge_index = []
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if i != j and similarity_matrix[i, j] > threshold:
                edge_index.append([i, j])
    if len(edge_index) == 0:
        return torch.empty(2, 0, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def energy_propagation(energies, edge_index, prop_layers=1, alpha=0.5, prop_weights=None):
    '''Energy belief propagation, returning the propagated energy.'''
    e = torch.tensor(energies).float()

    N = e.shape[0]
    row, col = edge_index

    if prop_weights is None:
        prop_weights = torch.ones(N).float()

    d = degree(col, N).float()
    d_norm = 1. / d[col]
    value = torch.ones_like(row) * d_norm * prop_weights[col]
    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    adj = SparseTensor(row=col, col=row, value=value, sparse_sizes=(N, N))

    for _ in range(prop_layers):
        e = e * alpha + matmul(adj, e) * (1 - alpha)

    return e.cpu().numpy()


def fed_EnergyBelief(global_model, selected_models, client_energies, args):
    """
    Implement EnergyBelief: A robust aggregation method based on energy propagation

    参数:
    - global_model: global model
    - selected_models: List of selected client models
    - client_energies: List of energy differences for each client
    - args: Parameter object, including threshold tau, number of propagation layers prop_layers, propagation coefficient prop_alpha, etc
    """

    similarity_matrix = compute_similarity_matrix(client_energies)
    print("Similarity Matrix:\n", similarity_matrix)
    edge_index = build_edge_index(similarity_matrix, threshold=args.tau)
    excluded_clients = set()
    for i, similarities in enumerate(similarity_matrix):
        if np.all(similarities[np.arange(len(similarity_matrix)) != i] < args.tau):
            excluded_clients.add(i)
    selected_models_filtered = [model for i, model in enumerate(selected_models) if i not in excluded_clients]
    client_energies_filtered = [energy for i, energy in enumerate(client_energies) if i not in excluded_clients]

    included_indices = [i for i in range(len(client_energies)) if i not in excluded_clients]
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(included_indices)}
    edge_index_filtered = []
    for edge in edge_index.t().tolist():
        if edge[0] in excluded_clients or edge[1] in excluded_clients:
            continue
        edge_index_filtered.append([index_mapping[edge[0]], index_mapping[edge[1]]])
    if len(edge_index_filtered) > 0:
        edge_index_filtered = torch.tensor(edge_index_filtered, dtype=torch.long).t().contiguous()
    else:
        edge_index_filtered = torch.empty(2, 0, dtype=torch.long)

    propagation_weights = np.zeros(len(selected_models_filtered))
    total_edges = edge_index_filtered.shape[1] if edge_index_filtered.numel() > 0 else 0
    if total_edges > 0:
        for i in range(len(selected_models_filtered)):
            propagation_weights[i] = (edge_index_filtered[1] == i).sum().item()
        propagation_weights /= total_edges

    propagated_energies = energy_propagation(
        client_energies_filtered,
        edge_index_filtered,
        prop_layers=args.prop_layers,
        alpha=args.prop_alpha,
        prop_weights=torch.tensor(propagation_weights)
    )

    propagated_energies = np.mean(propagated_energies, axis=1)
    inverted_energies = -propagated_energies
    combined_weights = inverted_energies * propagation_weights

    if combined_weights.sum() == 0:
        combined_weights = np.ones(len(combined_weights)) / len(combined_weights)
    else:
        combined_weights /= combined_weights.sum()

    global_params = global_model.state_dict()
    for param_tensor in global_params.keys():
        model_params = [model.state_dict()[param_tensor].cpu() for model in selected_models_filtered]
        shapes = [param.shape for param in model_params]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Shape mismatch in parameter {param_tensor}: {shapes}")
        avg = sum(param * weight for param, weight in zip(model_params, combined_weights))
        global_params[param_tensor].copy_(avg)
    global_model.load_state_dict(global_params)
    return global_model

def compute_cosine_similarities(local_models):
    num_clients = len(local_models)
    similarities = torch.zeros((num_clients, num_clients))

    client_params = [torch.cat([param.view(-1) for param in model.parameters()]) for model in local_models]

    for i in range(num_clients):
        for j in range(num_clients):
            if i != j:
                similarities[i, j] = torch.nn.functional.cosine_similarity(client_params[i], client_params[j], dim=0)
            else:
                similarities[i, j] = 1.0

    return similarities


