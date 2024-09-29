import numpy as np
import torch
from finch import FINCH
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from random import sample

def select_models_based_on_energy(client_energies):
    """
    Cluster the models based on energy differences to identify malicious clients.
    args:
    - client_energies: List of energy differences for each client
    Return:
    - selected_models_index: List of selected model indices
    """
    max_length = max(len(energy) for energy in client_energies)
    padded_energies = np.array(
        [np.pad(energy, (0, max_length - len(energy)), 'constant') for energy in client_energies])
    c, num_clust, req_c = FINCH(padded_energies)
    labels = c[:, 0]
    cluster_centers = [padded_energies[labels == i].mean(axis=0) for i in range(num_clust[0])]
    if num_clust[0] == 1:
        selected_models_index = sample(range(len(client_energies)), 7)
    else:
        malicious_cluster = np.argmax(np.sum(cluster_centers, axis=1))
        selected_models_index = [i for i, label in enumerate(labels) if label != malicious_cluster]

    return selected_models_index



