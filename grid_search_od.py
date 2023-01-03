import numpy as np
import pickle as pkl
import argparse

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
from pyod.models.abod import ABOD
from pyod.models.copod import COPOD
from pyod.models.cof import COF
from pyod.models.ecod import ECOD
from pyod.models.inne import INNE
from pyod.models.iforest import IForest

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

def parse_args():
    parser = argparse.ArgumentParser(description='Grid search for outlier detection script')
    parser.add_argument('-t', type=str, help='Type of outlier detection technique')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    syn_folder = Path('datasets', 'synthetic')
    rw_folder = Path('datasets', 'real-world')
    rw_datasets = {x.stem for x in rw_folder.iterdir() if x.is_file()}
    
    filename_mapper = {
        'lof': 'lof',
        'abod-default': 'abod_default',
        'abod-fast': 'abod_fast',
        'copod': 'copod',
        'cof': 'cof',
        'ecod': 'ecod',
        'inne': 'inne',
        'iforest': 'iforest',
        'knn': 'knn',
        'hbos': 'hbos'
    }
    
    result_path = Path('other_outlier_det')
    savepath = result_path / f'{filename_mapper[args.t]}_simpler3.pkl'
    # k = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 45, 50]
    k = [3, 4, 5, 7, 10, 12, 15]
    
    print('@' * 50)
    print(f'Savepath: {str(savepath)}')
    print('@' * 50)

    data_list = []
    for fullpath in syn_folder.iterdir():
        if fullpath.is_file() and fullpath.suffix != '.ipynb':
            data_list.append(fullpath)
    for fullpath in rw_folder.iterdir():
        if fullpath.is_file() and fullpath.suffix != '.ipynb':
            data_list.append(fullpath)

    if not result_path.is_dir():
        result_path.mkdir()

    syn_cache = DatasetBridgeCache(syn_folder)
    rw_cache = DatasetBridgeCache(rw_folder)

    contamination_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, .1, .2, .3, .4, .5]
    n_neigh_range = [3, 5, 7, 10, 12, 15, 20]
    if args.t == 'lof':
        parameter_dict = {
            LOF: {
               'contamination': contamination_range,
               'n_neighbors': n_neigh_range,
               'n_jobs': [2]
            }
        }
    elif args.t == 'knn':
        parameter_dict = {
            KNN: {
                'contamination': contamination_range,
                'n_neighbors': n_neigh_range,
                'n_jobs': [2]
            }
        }
    elif args.t == 'hbos':
        parameter_dict = {
            HBOS: {
                'n_bins': [10, 15, 20, 25, 30, 35, 40, 45, 50],
                'tol': [.1, .2, .3, .4, .5],
                'contamination': contamination_range,
            },
        }
    elif args.t == 'abod-fast':
        parameter_dict = {
            ABOD: {
                'contamination': contamination_range,
                'n_neighbors': n_neigh_range,
                'method': ['fast']
            },
        }
    elif args.t == 'abod-default':
        parameter_dict = {
            ABOD: {
                'contamination': contamination_range,
                'n_neighbors': n_neigh_range,
                'method': ['default']
            },
        }
    elif args.t == 'copod':
        parameter_dict = {
            COPOD: {
                'contamination': contamination_range,
                'n_jobs': [2]
            },
        }
    elif args.t == 'cof':
        parameter_dict = {
            COF: {
                'contamination': contamination_range,
                'n_neighbors': n_neigh_range
            },
        }
    elif args.t == 'ecod':
        parameter_dict = {
            ECOD: {
                'contamination': contamination_range,
                'n_jobs': [2]
            },
        }
    elif args.t == 'inne':
        parameter_dict = {
            INNE: {
                'contamination': contamination_range,
                'n_estimators': [100, 200, 300, 400, 500],
                'max_samples': [8, 'auto', 0.1, 0.05, 0.15],
                'random_state': [47]
            },
        }
    elif args.t == 'iforest':
        # parameter_dict = {
        #     IForest: {
        #         'contamination': [.1, .2, .3, .4, .5],
        #         'n_estimators': [100, 250, 500],
        #         'max_samples': [256, 'auto', 0.1, 0.2],
        #         # 'max_features': [1.0, 0.8],
        #         'bootstrap': [True, False],
        #         'behaviour': ['new', 'old'],
        #         'n_jobs': [2]
        #     },
        # }
        parameter_dict = {
            IForest: {
                'contamination': contamination_range,
                'n_estimators': [100, 200, 300, 400, 500],
                'max_samples': [256, 'auto', 128, 0.05, 0.1, 0.15],
                # 'max_features': [1.0, 0.8],
                'bootstrap': [True, False],
                'behaviour': ['new', 'old'],
                'n_jobs': [2]
            },
        }
    else:
        raise ValueError(f'Invalid t parameter {args.t}!')
        
    # parameter_dict = {
        # LOF: {
        #    'contamination': contamination_range,
        #    'n_neighbors': n_neigh_range,
        #    'n_jobs': [2]
        # },
        # ABOD: {
        #     'contamination': contamination_range,
        #     'n_neighbors': n_neigh_range,
        #     'method': ['fast']
        # },
        # ABOD: {
        #     'contamination': contamination_range,
        #     'n_neighbors': n_neigh_range,
        #     'method': ['default']
        # },
        # COPOD: {
        #     'contamination': contamination_range,
        #     'n_jobs': [2]
        # },
        # COF: {
        #     'contamination': contamination_range,
        #     'n_neighbors': n_neigh_range
        # },
        # ECOD: {
        #     'contamination': contamination_range,
        #     'n_jobs': [2]
        # },
        # INNE: {
        #     'contamination': contamination_range,
        #     'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        #     'max_samples': [8, 'auto', 0.1, 0.05, 0.15],
        #     'random_state': [47]
        # },
        # IForest: {
        #     'contamination': contamination_range,
        #     'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
        #     'max_samples': [256, 'auto', 128, 512, 0.05, 0.1, 0.15],
        #     'max_features': [1.0, 0.8],
        #     'bootstrap': [True, False],
        #     'behaviour': ['new', 'old'],
        #     'n_jobs': [2]
        # },
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
    # }

    #dataset name -> k -> model name -> list of tuples: (parameter dict, dict{key, (confusion matrix, precision array, recall array, f1 array)}, 
    #                                                       keys: 'bridge_vs_outlier', 'bridge+transition_vs_outliers', 'bridge_vs_outliers+transition', 'bridge+transition_vs_outliers+transition', 'gtcore_vs_predcore')
    # outlier detection
    result_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    print(result_path)

    for k_val in k:
        print(f'Analysing k value: {k_val}')
        for fullpath in data_list:
            dt = fullpath.stem + fullpath.suffix
            dt_no_ext = fullpath.stem
            print('###################################')
            print('Analysing dataset %s' % dt)

            cache = rw_cache if dt_no_ext in rw_datasets else syn_cache
            X, cluster_labels = cache.get_data(dt_no_ext, suffix=fullpath.suffix[1:])
            local_distances, local_indices = cache.get_neighbor(dt_no_ext, k_val, suffix=fullpath.suffix[1:])
            gt_is_bridge = cache.get_bridge(dt, k_val, suffix=fullpath.suffix[1:])

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



