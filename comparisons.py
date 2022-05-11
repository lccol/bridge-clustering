import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from typing import Dict, Union
from sklearn.cluster import DBSCAN, OPTICS
from hdbscan import HDBSCAN
from border_peel.border_peeling import BorderPeelingWrapper, BorderPeel
from bridge_clustering.functions import determine_bridges, compute_neighbors
from bridge_clustering import BridgeClustering
from autoclust import AUTOCLUST
from denmune_wrapper import DenMuneWrapper

from pyod.models.lof import LOF
from utils import read_arff, generate_dbscan_config_tree
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, confusion_matrix, recall_score
from utils import compute_recall
from statistical_tests import friedman_test, generate_reports, nemenyi_test

def clusterize(kls, args, X, dbscan_config: Dict, dataset: str) -> np.ndarray:
    if kls == DBSCAN:
        args2 = copy.copy(args)
        min_points = args['min_samples']
        eps = dbscan_config[dataset][min_points]
        args2['eps'] = eps
    elif kls == BridgeClustering:
        args2 = copy.copy(args)
        outlier_kls = args['outlier_class']
        outlier_params = args['outlier_params']
        outlier_clf = outlier_kls(**outlier_params)
        del args2['outlier_class']
        del args2['outlier_params']
        args2['outlier_detection'] = outlier_clf
    else:
        args2 = args
    clf = kls(**args2)
    clf.fit(X)
    return clf

def generate_figure(X, cluster_label, save_path=None, plot_config=None):
    if plot_config is None:
        plot_config = {}

    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=cluster_label, marker='.', alpha=.5)
    plt.axis('off')
    if not save_path is None:
        fig.savefig(save_path, **plot_config)
    return fig

def check_recall(is_bridge: np.ndarray, outliers: np.ndarray) -> float:
    cm = confusion_matrix(is_bridge, outliers, labels=[0, 1])
    my_recall = compute_recall(cm, zero_div=-2)[1]
    recall = recall_score(is_bridge, outliers, labels=[0, 1], pos_label=1)

    if my_recall >= 0:
        assert (my_recall - recall) < 1e-5

    return my_recall

if __name__ == '__main__':
    dataset_basepath = Path('datasets')
    dbscan_config_path = Path('dbscan_config.json')
    export_images = True
    export_images_path = Path('results', 'clustering_comparisons_figures')
    report_basepath = Path('results', 'reports')
    k = 10

    if not export_images_path.is_dir():
        export_images_path.mkdir(parents=True)

    if not report_basepath.is_dir():
        report_basepath.mkdir(parents=True)

    plot_configuration = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }

    dbscan_configuration = generate_dbscan_config_tree(dbscan_config_path)

    classifiers = [
        (DBSCAN, {'min_samples': 7}), 
        (HDBSCAN, {'cluster_selection_method': 'eom', 'min_cluster_size': 45, 'min_samples': 5}),
        (OPTICS, {'min_samples': 50, 'cluster_method': 'xi'}),
        (BridgeClustering, {'outlier_class': LOF, 'outlier_params': {'contamination': .2, 'n_neighbors': 15}, 'k': k}),
        (BorderPeelingWrapper, {}),
        (AUTOCLUST, {}),
        (DenMuneWrapper, {'k': 20, 'rgn_tsne': False, 'show_plots': False, 'show_noise': True, 'show_analyzer': False})
    ]

    df_dict = defaultdict(list)
    recall_df_dict = defaultdict(list)

    for fullpath in dataset_basepath.iterdir():
        dataset_filename_no_ext = fullpath.stem
        dataset = f'{dataset_filename_no_ext}.arff'
        print('@' * 30)
        print(f'dataset: {dataset}')
        df_dict['dataset'].append(dataset)

        X, labels = read_arff(fullpath)
        _, indices = compute_neighbors(X, k)
        is_bridge = determine_bridges(labels, indices) == 0
        for kls, args in classifiers:
            print(f'Clustering algorithm: {str(kls)}')
            clf = clusterize(kls, args, X, dbscan_configuration, dataset)
            predicted_labels = clf.labels_
            ari = adjusted_rand_score(labels, predicted_labels)
            
            df_dict[kls.__name__].append(ari)

            if kls == BridgeClustering:
                outliers = clf.outliers_
                recall = check_recall(is_bridge, outliers)
                recall_df_dict['ari'].append(ari)
                recall_df_dict['dataset'].append(dataset)
                recall_df_dict['recall'].append(recall)

            if export_images:
                savepath = export_images_path / f'{dataset_filename_no_ext}_{kls.__name__}.png'
                fig = generate_figure(X, predicted_labels, savepath, plot_configuration)
                plt.close(fig)
        if export_images:
            savepath = export_images_path / f'{dataset_filename_no_ext}_gt.png'
            fig = generate_figure(X, labels, savepath, plot_configuration)
            plt.close(fig)

    df = pd.DataFrame(df_dict).set_index('dataset').sort_index()
    df.to_csv(report_basepath / 'ari_scores.csv')
    recall_df = pd.DataFrame(recall_df_dict).set_index('dataset').sort_index()
    recall_df.to_csv(report_basepath / 'ari_vs_recall.csv')

    print('Mean ARI:')
    print(df.mean())
    print('@' * 30)
    print('Median ARI:')
    print(df.median())

    generate_reports(df, report_basepath, save_df=False, save_rankings_df=True)