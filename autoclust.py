import networkx as nx
import numpy as np
from random import Random
from numpy.core.fromnumeric import mean
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean

class Edge:
    def __init__(self, src: int, dst: int, weight: float):
        self.src = src
        self.dst = dst
        self.weight = weight

    @classmethod
    def tuple_representation(self, t):
        first, last = min(t[:2]), max(t[:2])
        return (first, last, *t[2:])
    
    def get_tuple_representation(self):
        t = (self.src, self.dst)
        return self.tuple_representation(t)

    @classmethod
    def convert_weight_dict(self, t):
        d = t[-1]
        w = d['weight']
        return (t[0], t[1], w)
        

class AUTOCLUST:
    def __init__(self, cache_distances: bool=True, eval_empty_nodes: bool=False, shuffle_labels: bool=True, seed=None):
        self.long_edges_ = []
        self.short_edges_ = []
        self.other_edges = []
        self.local_mean = []
        self.local_mean2 = []
        self.std_devs = []
        self.invalid = []
        self.second_order_local_mean = []
        self.labels_ = None
        self.eval_empty_nodes = eval_empty_nodes
        self.g = nx.Graph()

        self.mean_std_dev = -1
        self.delaunay_ = None
        self.cluster_sizes = {}

        self.cache_distances = cache_distances
        self.shuffle_labels = shuffle_labels
        self.seed = seed
        if seed is None:
            self.rand = Random()
        else:
            self.rand = Random(seed)
        self.dist_cache_ = None if not cache_distances else {}
        return

    def fit(self, X, y=None):
        self.delaunay_ = Delaunay(X)
        tri = self.delaunay_.simplices
        self.labels_ = np.ones(X.shape[0], dtype=int) * -2

        self._add_edges(tri, X)
        self._compute_stats(X.shape[0])
        self._label_edges(X.shape[0])

        self._remove_edges(self.short_edges_)
        self._remove_edges(self.long_edges_)

        
        self._phase_1()
        self._phase_2(X.shape[0])
        self._phase_3(X.shape[0])

        return self

    def fit_predict(self, X, y=None):
        self.fit(X, y)
        return self.labels_

    def _add_edges(self, tri, X):
        for t in tri:
            for idx, u in enumerate(t):
                v = t[idx + 1] if idx < (len(t) - 1) else t[0]
                w = self.compute_edge_weight(u, v, X)

                self.g.add_edge(u, v, weight=w)

        for node in range(X.shape[0]):
            if not self.g.has_node(node):
                self.g.add_node(node)

        return

    def _label_conn_comp(self, shuffle_labels: bool=False):
        cc = list(nx.connected_components(self.g))
        if shuffle_labels:
            self.rand.shuffle(cc)
        labels = np.ones_like(self.labels_, dtype=int) * -2
        sizes = {}
        for idx, c in enumerate(cc):
            sizes[idx] = len(c)
            for node in c:
                labels[node] = idx
        return labels, sizes

    def _remove_edges(self, to_remove):
        for edges in to_remove:
            for edge in edges:
                if self.g.has_edge(edge.src, edge.dst):
                    self.g.remove_edge(edge.src, edge.dst)
        return

    def _compute_stats(self, npoints):
        for node in range(npoints):
            edges = list(self.g.edges(node, data=True))
            is_invalid = len(edges) == 0 if not self.eval_empty_nodes else False
            weights = np.array([d['weight'] for _, _, d in edges]) if not is_invalid else 0
            mean_weight = weights.mean() if not is_invalid else 0
            std_dev = weights.std() if not is_invalid else 0

            self.local_mean.append(mean_weight)
            self.std_devs.append(std_dev)
            self.invalid.append(is_invalid)
            
        self.local_mean = np.ma.masked_array(self.local_mean, mask=self.invalid)
        self.std_devs = np.ma.masked_array(self.std_devs, mask=self.invalid)

        self.mean_std_dev = self.std_devs.mean()
        return

    def _label_edges(self, npoints):
        for node in range(npoints):
            edges = list(self.g.edges(node, data=True))
            short = []
            other = []
            long = []
            for edge in edges:
                u, v, d = edge
                assert u == node
                w = d['weight']
                if w < (self.local_mean[node] - self.mean_std_dev):
                    short.append(Edge(u, v, w))
                elif w > (self.local_mean[node] + self.mean_std_dev):
                    long.append(Edge(u, v, w))
                else:
                    other.append(Edge(u, v, w))
            self.short_edges_.append(short)
            self.other_edges.append(other)
            self.long_edges_.append(long)
        return

    def compute_edge_weight(self, u, v, X):
        if not self.cache_distances:
            return euclidean(X[u], X[v])
        else:
            k = (min(u, v), max(u, v))
            if k in self.dist_cache_:
                return self.dist_cache_[k]
            else:
                d = euclidean(X[u], X[v])
                self.dist_cache_[k] = d
                return d

    def _phase_1(self):
        self.labels_, self.cluster_sizes = self._label_conn_comp()
        return

    def _phase_2(self, npoints):
        edges_to_add = []
        for node in range(npoints):
            shorts = self.short_edges_[node]

            max_cc = -100
            max_cc_size = -2
            max_minweight = -2
            for e in shorts:
                dst_label = self.labels_[e.dst]
                if self.cluster_sizes[dst_label] > max_cc_size or (self.cluster_sizes[dst_label] == max_cc_size and (e.weight < max_minweight or max_minweight == -2)):
                    max_minweight = e.weight
                    if max_cc != dst_label:
                        max_cc = dst_label
                        max_cc_size = self.cluster_sizes[dst_label]
                        edges_to_add.clear()
                    else:
                        assert max_cc_size == self.cluster_sizes[dst_label]
                    edges_to_add.append(e)
            
        for e in edges_to_add:
            self.g.add_edge(e.src, e.dst, weight=e.weight)

        self.labels_, self.cluster_sizes = self._label_conn_comp()
        return

    def _phase_3(self, npoints):
        normalized_edges_dict = {}
        to_remove = set()
        for node in range(npoints):
            # transforms int, int, dict into int, int, float (src, dst, weight)
            tmp = set(Edge.tuple_representation(Edge.convert_weight_dict(t)) for t in self.g.edges(node, data=True))
            normalized_edges_dict[node] = tmp
        for node in range(npoints):
            local_edges = normalized_edges_dict[node].copy()
            edges2 = local_edges
            for neigh in local_edges:
                other = neigh[1]
                other_edges = normalized_edges_dict[other]
                edges2 = edges2.union(other_edges)
            mean2 = mean(list(map(lambda e: e[-1], edges2))) if len(edges2) > 0 else 0
            self.local_mean2.append(mean2)

            for e in edges2:
                w = e[-1]
                if w > (mean2 + self.mean_std_dev):
                    to_remove.add((e[0], e[1]))

        self.local_mean2 = np.array(self.local_mean2)
        for e in to_remove:
            self.g.remove_edge(e[0], e[1])

        self.labels_, self.cluster_sizes = self._label_conn_comp(self.shuffle_labels)
        return