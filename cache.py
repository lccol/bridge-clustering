import os
from collections import defaultdict

from bridge_clustering.functions import compute_neighbors, determine_bridges
from utils import read_arff

class DatasetBridgeCache:
    def __init__(self, path) -> None:
        self.path = path

        self.data_cache = {}
        self.bridge_cache = defaultdict(dict)
        self.neighbor_cache = defaultdict(dict)

    def append_arff(self, dataset):
        if not dataset.endswith('.arff'):
            dataset = dataset + '.arff'
        return dataset

    def get_data(self, dataset):
        dataset = self.append_arff(dataset)
        if not dataset in self.data_cache:
            X, cluster_labels = read_arff(os.path.join(self.path, dataset))
            self.data_cache[dataset] = (X, cluster_labels)

        return self.data_cache[dataset]

    def get_neighbor(self, dataset, k):
        dataset = self.append_arff(dataset)
        if not dataset in self.neighbor_cache or not k in self.neighbor_cache[dataset]:
            X, _ = self.get_data(dataset)
            local_distances, local_indices = compute_neighbors(X, k)
            self.neighbor_cache[dataset][k] = (local_distances, local_indices)
        return self.neighbor_cache[dataset][k]

    def get_bridge(self, dataset, k):
        dataset = self.append_arff(dataset)
        if not dataset in self.bridge_cache or not k in self.bridge_cache[dataset]:
            _, cluster_labels = self.get_data(dataset)
            _, local_indices = self.get_neighbor(dataset, k)

            is_bridge = determine_bridges(cluster_labels, local_indices) == 0
            self.bridge_cache[dataset][k] = is_bridge
        return self.bridge_cache[dataset][k]