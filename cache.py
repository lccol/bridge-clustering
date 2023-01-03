import os
from collections import defaultdict

from bridge_clustering.functions import compute_neighbors, determine_bridges
from utils import read_arff, read_dataset

class DatasetBridgeCache:
    def __init__(self, path) -> None:
        self.path = path

        self.data_cache = {}
        self.bridge_cache = defaultdict(dict)
        self.neighbor_cache = defaultdict(dict)

    def append_arff(self, dataset, suffix='arff'):
        if not dataset.endswith(f'.{suffix}'):
            dataset = dataset + f'.{suffix}'
        return dataset

    def get_data(self, dataset, suffix='arff'):
        dataset = self.append_arff(dataset, suffix)
        if not dataset in self.data_cache:
            X, cluster_labels = read_dataset(os.path.join(self.path, dataset))
            self.data_cache[dataset] = (X, cluster_labels)

        return self.data_cache[dataset]

    def get_neighbor(self, dataset, k, suffix='arff'):
        dataset = self.append_arff(dataset, suffix)
        if not dataset in self.neighbor_cache or not k in self.neighbor_cache[dataset]:
            X, _ = self.get_data(dataset, suffix)
            local_distances, local_indices = compute_neighbors(X, k)
            self.neighbor_cache[dataset][k] = (local_distances, local_indices)
        return self.neighbor_cache[dataset][k]

    def get_bridge(self, dataset, k, suffix='arff'):
        dataset = self.append_arff(dataset, suffix)
        if not dataset in self.bridge_cache or not k in self.bridge_cache[dataset]:
            _, cluster_labels = self.get_data(dataset, suffix)
            _, local_indices = self.get_neighbor(dataset, k, suffix)

            is_bridge = determine_bridges(cluster_labels, local_indices) == 0
            self.bridge_cache[dataset][k] = is_bridge
        return self.bridge_cache[dataset][k]