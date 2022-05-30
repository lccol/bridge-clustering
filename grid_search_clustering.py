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

from denmune_wrapper import DenMuneWrapper

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
    savepath = result_path / f'hdbscan2.pkl'
    dbscan_config_path = Path('dbscan_config.json')
    eps_dict = generate_dbscan_config_tree(dbscan_config_path)

    if not result_path.is_dir():
        result_path.mkdir()

    dataset_cache = DatasetBridgeCache(dataset_folder)

    parameter_dict = {
#         OPTICS: {
#             'min_samples': [0.01, 0.03, 0.05, 0.07, 0.1, 5, 50, 100],
#             'cluster_method': ['xi', 'dbscan'],
#             'n_jobs': [2]
#         }
        HDBSCAN: {
           'min_cluster_size': [5, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150],
           'min_samples': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
           'cluster_selection_method': ['eom', 'leaf']
        }
#         DBSCAN: {
#            'min_samples': [5, 7, 10, 12, 15, 17]
#         },
#         DenMuneWrapper: {
#             'k': list(range(5, 71, 5)) + [6, 39],
#             'rgn_tsne': [False],
#             'show_plots': [False],
#             'show_noise': [True],
#             'show_analyzer': [False]
#         }
    }


    # clustering
    result_dict = defaultdict(lambda: defaultdict(list))
    
    print(result_path)

    for fullpath in dataset_folder.iterdir():
        dt = fullpath.stem + fullpath.suffix
        dt_no_ext = fullpath.stem
        print('###################################')
        print('Analysing dataset %s' % dt)

        X, cluster_labels = dataset_cache.get_data(dt_no_ext)

        for model in parameter_dict:
            args = parameter_dict[model]

            print('___________________________________')
            print(f'analysing dataset {dt_no_ext} with model {model.__name__}')

            for p in parameter_generator(args):
                print(f'evaluating parameters {p}')
                # DBScan
                if isinstance(model, DBSCAN):
                    k = p['min_samples']
                    eps = eps_dict[dt][k]
                    p['eps'] = eps

                od = model(**p)
                od.fit(X)

                # clustering
                predicted_labels = od.labels_

                score_dict = {}
                ari = adjusted_rand_score(cluster_labels, predicted_labels)
                score_dict['ari'] = ari

                # clustering
                result_dict[dt_no_ext][model.__name__].append((p, score_dict))

            print(f'end of analysis for model {model.__name__}')

        with open(savepath, 'wb') as fp:
            pkl.dump(convert_to_dict(result_dict), fp)

        print(f'terminated analysis for dataset {dt_no_ext}')



