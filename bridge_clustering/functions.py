import heapq
import numpy as np

from typing import  Union
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors

from collections import defaultdict

def generate_rknn(local_indices):
    rknn = defaultdict(list)
    for i, neighbors in enumerate(local_indices):
        for neigh in neighbors:
            rknn[neigh].append(i)

    return dict(rknn)

def check_jump_matrix(jump_matrix: np.ndarray, local_indices: np.ndarray, is_bridge_candidate: np.ndarray):
    npoints = is_bridge_candidate.size
    rknn = generate_rknn(local_indices)

    indexes = np.arange(npoints, dtype=np.int)
    bridge_indexes = indexes[is_bridge_candidate]
    bridge_set = set(bridge_indexes.tolist())
    for i in range(npoints):
        if is_bridge_candidate[i]:
            assert jump_matrix[:, i].sum() == 0
            assert jump_matrix[i, :].sum() == 0
            continue
        rknn_set = set(rknn[i])
        knn_set = set(local_indices[i])
        union = rknn_set.union(knn_set)

        union = union.difference(bridge_set)

        jumps = set(indexes[jump_matrix[i]])
        assert jumps == union
    print('check on jump matrix passed')
    return


def expand_labels(local_indices, is_bridge_candidate):
    n_points = local_indices.shape[0]
    cluster_labels = np.ones(n_points) * -1
    bridge_indexes = np.arange(n_points)[is_bridge_candidate]
    x_axis = np.arange(n_points)[..., np.newaxis]

    jump_matrix = np.zeros((n_points, n_points), dtype=bool)
    
    assert ((local_indices == x_axis) == False).all()

    jump_matrix[x_axis, local_indices] = True
    jump_matrix[x_axis, bridge_indexes] = False

    transposed_jump_matrix = jump_matrix.T
    jump_matrix = jump_matrix | transposed_jump_matrix

    jump_matrix[is_bridge_candidate, :] = False
    jump_matrix[:, is_bridge_candidate] = False

    assert (np.diag(jump_matrix) == False).all()
    # check_jump_matrix(jump_matrix, local_indices, is_bridge_candidate)

    index_line = np.arange(n_points)

    label = 0
    for i in range(n_points):
        if is_bridge_candidate[i]:
            continue
        if cluster_labels[i] != -1:
            continue
        pq = [i]

        while pq:
            el = pq.pop()
            cluster_labels[el] = label
            elements_to_explore = index_line[jump_matrix[el] & (cluster_labels == -1)].tolist()
            pq.extend(elements_to_explore)

        label += 1

    return cluster_labels

def predict_and_cluster(clf, X: np.ndarray, k: Union[None, int], local_indices: Union[np.ndarray, None]=None, local_distances: Union[np.ndarray, None]=None):
    clf.fit(X)
    pred_bridges = clf.labels_ == 1
    return compute_cluster_labels(X, k, pred_bridges, local_indices, local_distances), pred_bridges

def generate_jump_distance_matrix(local_distances, local_indices):
    npoints = local_indices.shape[0]
    jump_matrix = np.zeros((npoints, npoints), dtype=np.float)

    index_array = np.arange(npoints, dtype=np.int)
    jump_matrix[index_array[..., np.newaxis], local_indices[np.newaxis, ...]] = local_distances

    mask = jump_matrix == 0
    transposed_jump_matrix = jump_matrix.T
    jump_matrix = jump_matrix + transposed_jump_matrix * mask

    assert (jump_matrix == jump_matrix.T).all()
    return jump_matrix

def verify_jump_distance_matrix(local_distances, local_indices, jump_distance_matrix):
    npoints = local_indices.shape[0]
    result = np.zeros((npoints, npoints))
    for i in range(npoints):
        for j, dist in zip(local_indices[i], local_distances[i]):
            result[i, j] = dist
            result[j, i] = dist

    assert (result == jump_distance_matrix).all()
    return

def assign_bridge_labels(cluster_labels, local_indices, local_distances):
    def _collect_data(jump_matrix, is_bridge_candidate, index):
        not_ignore = jump_matrix > 0
        is_bridge_candidate = is_bridge_candidate & not_ignore
        not_bridge_candidate = (~is_bridge_candidate) & not_ignore
        neigh_indexes = np.arange(is_bridge_candidate.size, dtype=np.int)
        index_list = np.ones_like(neigh_indexes) * index

        assert (cluster_labels[neigh_indexes[is_bridge_candidate]] == -1).all()
        assert (cluster_labels[neigh_indexes[not_bridge_candidate]] != -1).all()

        assert (jump_matrix[not_bridge_candidate] > 0).all()
        assert (jump_matrix[is_bridge_candidate] > 0).all()

        valid_queue = list(zip(jump_matrix[not_bridge_candidate], neigh_indexes[not_bridge_candidate], index_list))
        invalid_queue = list(zip(jump_matrix[is_bridge_candidate], neigh_indexes[is_bridge_candidate], index_list))

        return valid_queue, invalid_queue

    index_array = np.arange(cluster_labels.size)
    is_bridge_candidate = cluster_labels == -1
    missing_indexes = index_array[is_bridge_candidate]
    jump_distance_matrix = generate_jump_distance_matrix(local_distances, local_indices)
    # verify_jump_distance_matrix(local_distances, local_indices, jump_distance_matrix)

    pq = []
    pq2 = defaultdict(list)
    for i in missing_indexes:
        jump_row = jump_distance_matrix[i]

        valid_queue, invalid_queue = _collect_data(jump_row, is_bridge_candidate, i)
        for t in invalid_queue:
            pq2[t[1]].append(t)

        pq.extend(valid_queue)

    heapq.heapify(pq)

    while pq:
        t = heapq.heappop(pq)
        neigh, me = int(t[1]), int(t[2])
        if cluster_labels[me] != -1:
            continue
        assert cluster_labels[neigh] != -1

        cluster_labels[me] = cluster_labels[neigh]
        to_add = list(filter(lambda x: cluster_labels[x[2]] == -1, pq2[me]))
        heapq.heapify(to_add)

        pq = list(heapq.merge(pq, to_add))
    return cluster_labels

def compute_cluster_labels(X: Union[None, np.ndarray], k: Union[None, int], is_bridge_candidate: np.ndarray, local_indices: Union[None, np.ndarray]=None, local_distances: Union[None, np.ndarray]=None):
    if X is None:
        assert k is None
        assert not local_indices is None
    if local_indices is None:
        local_distances, local_indices = compute_neighbors(X, k)
    assert is_bridge_candidate.dtype == 'bool'
    pred_labels = expand_labels(local_indices, is_bridge_candidate)
    pred_labels = assign_bridge_labels(pred_labels, local_indices, local_distances)
    return pred_labels

def determine_bridges(labels, local_indices):
    matrix = labels[local_indices]
    self_labels = labels[..., np.newaxis]

    bridges = (matrix == self_labels).all(axis=-1)
    return bridges

def remove_self(distances, indices):
    indexes = np.arange(indices.shape[0], dtype=np.int)[..., np.newaxis]
    mask = indices == indexes
    check_array = (mask.sum(axis=-1)) == 1
    check = check_array.all()
    if not check:
        print(f'WARNING: found at least one point for which the point itself is not included in its neighborhood.')
        print('This is likely due to overlapping points.')
        # for points for which distances == 0 for all the neighbors and for which the point itself is not included,
        # remove the first element
        null_elements = ~check_array
        null_indexes = indexes[null_elements]
        mask[null_indexes, 0] = True
    inverse_mask = ~mask

    test_dist = remove_first_column(distances)
    
    new_distances = distances[inverse_mask].reshape(test_dist.shape)
    new_indices = indices[inverse_mask].reshape(test_dist.shape)


    assert (test_dist == new_distances).all()

    return new_distances, new_indices

def remove_first_column(arr):
    return arr[:, 1:]

def compute_neighbors(X, k=None):
    new_k = X.shape[0] if k is None else k + 1
    nn = NearestNeighbors(n_neighbors=new_k, algorithm='ball_tree').fit(X)
    nn_distances, nn_indices = nn.kneighbors(X)
    nn_distances, nn_indices = remove_self(nn_distances, nn_indices)
    return nn_distances, nn_indices