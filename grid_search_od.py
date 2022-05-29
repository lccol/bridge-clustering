import numpy as np
import pickle as pkl

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, adjusted_rand_score

from pathlib import Path

from typing import Union
from bridge_clustering.functions import expand_labels, assign_bridge_labels
from utils import read_arff, generate_dbscan_config_tree
from collections import defaultdict
from utils import compute_f1, compute_precision, compute_recall, parameter_generator
from cache import DatasetBridgeCache

from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN

from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN

def generate_results(gt, pred):
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    my_f1 = compute_f1(cm)
    my_precision = compute_precision(cm)
    my_recall = compute_recall(cm)

    prec, rec, f1, _ = precision_recall_fscore_support(gt, pred, labels=[0, 1], average=None)
    assert ((my_precision - prec) < 1e-3).all()
    assert ((my_recall - rec) < 1e-3).all()
    assert ((my_f1 - f1) < 1e-3).all()

    return cm, prec, rec, f1

def convert_to_dict(data):
    if isinstance(data, defaultdict):
        data = {k: convert_to_dict(v) for k, v in data.items()}

    return data

def clusterize(local_distances: np.ndarray, local_indices: np.ndarray, is_bridge_candidate: np.ndarray) -> np.ndarray:
    pred_cluster_labels = expand_labels(local_indices, is_bridge_candidate)
    pred_cluster_labels = assign_bridge_labels(pred_cluster_labels, local_indices, local_distances)
    return pred_cluster_labels

if __name__ == '__main__':
    dataset_folder = Path('datasets2')
    result_path = Path('grid_search_results2')
    savepath = result_path / f'lof2.pkl'
    k = [3, 4, 5, 7, 10, 12, 15]

    if not result_path.is_dir():
        result_path.mkdir()

    dataset_cache = DatasetBridgeCache(dataset_folder)

    contamination_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, .1, .2, .3, .4, .5]
    n_neigh_range = [3, 5, 7, 10, 12, 15, 20]
    parameter_dict = {
        LOF: {
           'contamination': contamination_range,
           'n_neighbors': n_neigh_range,
           'n_jobs': [2]
        },
#         HBOS: {
#            'n_bins': [10, 15, 20, 25, 30, 35, 40, 45, 50],
#            'tol': [.1, .2, .3, .4, .5],
#            'contamination': contamination_range,
#         },
#         KNN: {
#             'contamination': contamination_range,
#             'n_neighbors': n_neigh_range,
#             'n_jobs': [2]
#         }
    }

    #dataset name -> k -> model name -> list of tuples: (parameter dict, dict{key, (confusion matrix, precision array, recall array, f1 array)}, 
    #                                                       keys: 'bridge_vs_outlier', 'bridge+transition_vs_outliers', 'bridge_vs_outliers+transition', 'bridge+transition_vs_outliers+transition', 'gtcore_vs_predcore')
    # outlier detection
    result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    print(result_path)

    for k_val in k:
        print(f'Analysing k value: {k_val}')
        for fullpath in dataset_folder.iterdir():
            dt = fullpath.stem + fullpath.suffix
            dt_no_ext = fullpath.stem
            print('###################################')
            print('Analysing dataset %s' % dt)

            X, cluster_labels = dataset_cache.get_data(dt_no_ext)
            local_distances, local_indices = dataset_cache.get_neighbor(dt, k_val)
            gt_is_bridge = dataset_cache.get_bridge(dt, k_val)

            for model in parameter_dict:
                args = parameter_dict[model]
                
                print('___________________________________')
                print(f'analysing dataset {dt_no_ext} with model {model.__name__}, k {k_val}')

                for p in parameter_generator(args):
                    print(f'evaluating parameters {p}')

                    od = model(**p)
                    od.fit(X)

                    # outlier detection
                    pred_outliers = od.labels_ == 1

                    score_dict = {}

                    # outlier detection
                    # bridge vs outlier
                    cm, prec, rec, f1 = generate_results(gt_is_bridge, pred_outliers)
                    score_dict['bridge_vs_outlier'] = (cm, prec, rec, f1)
                    predicted_labels = clusterize(local_distances, local_indices, pred_outliers)

                    ari = adjusted_rand_score(cluster_labels, predicted_labels)
                    score_dict['ari'] = ari
                    # outlier detection
                    result_dict[dt_no_ext][k_val][model.__name__].append((p, score_dict))
                    

                print(f'end of analysis for model {model.__name__}')

            with open(savepath, 'wb') as fp:
                pkl.dump(convert_to_dict(result_dict), fp)

            print(f'terminated analysis for dataset {dt_no_ext}')



