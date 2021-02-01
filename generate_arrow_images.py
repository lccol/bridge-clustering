import os
import json
import numpy as np
import matplotlib.pyplot as plt

from utils import read_arff
from bridge_clustering.functions import compute_neighbors, determine_bridges, compute_cluster_labels
from pathlib import Path
from typing import Union, Set, List, Tuple
from pyod.models.lof import LOF

def compute_arrows(X: np.ndarray, neighbor_indices: Union[None, np.ndarray], k: Union[int, None], is_bridge: Union[None, np.ndarray]=None) -> np.ndarray:
    if neighbor_indices is None:
        assert not k is None
        _, neighbor_indices = compute_neighbors(X, k)

    if k is None:
        assert not neighbor_indices is None
        k = neighbor_indices.shape[1]

    if not is_bridge is None:
        indexes = np.arange(X.shape[0])
        bridge_indexes = indexes[is_bridge]
        bridge_neighbor_mask = (neighbor_indices[..., np.newaxis] == bridge_indexes[np.newaxis, np.newaxis, ...]).any(axis=-1)
        bridge_neighbor_mask_linear = bridge_neighbor_mask.flatten()

    scatter = X[neighbor_indices] # npoints x k x ndims

    new_X = np.expand_dims(X, axis=1)
    arrows = (scatter - new_X).reshape((-1, 2))
    if not is_bridge is None:
        arrows[bridge_neighbor_mask_linear] = 0
    centers = np.repeat(X, k, axis=0)

    return arrows, centers

def filter_bridges(centers: np.ndarray, arrows: np.ndarray, not_is_bridge: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    not_is_bridge_scattered = np.repeat(not_is_bridge, k, axis=0)

    return arrows[not_is_bridge_scattered], centers[not_is_bridge_scattered]

if __name__ == '__main__':
    dataset_path = Path('datasets')
    export_path = Path('figures', 'arrows_undir_DEF')

    plot_config = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }

    if not export_path.is_dir():
        export_path.mkdir(parents=True)
    
    k_list = [2, 5, 7, 10, 15, 20, 50]
    lof_params = {
        'n_neighbors': 15,
        'contamination': .2
    }
    figsize = (9, 9)

    for fullpath in dataset_path.iterdir():
        dt_no_ext = fullpath.stem

        X, cluster_labels = read_arff(fullpath)

        fig1 = plt.figure(figsize=figsize)
        plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, marker='.', alpha=.5)
        plt.axis('off')
        for k in k_list:
            distances, indices = compute_neighbors(X, k)
            is_bridge = determine_bridges(cluster_labels, indices) == 0
            not_bridge = ~is_bridge

            clf = LOF(**lof_params)
            clf.fit(X)
            outliers = clf.labels_ == 1
            not_outliers = ~outliers
            pred_labels = compute_cluster_labels(X, k, outliers, indices, distances)

            all_arrows, all_centers = compute_arrows(X, neighbor_indices=indices, k=k)
            arrows2, centers2 = compute_arrows(X, neighbor_indices=indices, k=k, is_bridge=is_bridge)
            filtered_arrows2, filtered_centers2 = filter_bridges(centers2, arrows2, not_bridge, k)

            pred_arrows, pred_centers = compute_arrows(X, neighbor_indices=indices, k=k, is_bridge=outliers)
            filtered_pred_arrows, filtered_pred_centers = filter_bridges(pred_centers, pred_arrows, not_outliers, k)

            fig2 = plt.figure(figsize=figsize)
            plt.scatter(X[:, 0][not_bridge], X[:, 1][not_bridge], c=cluster_labels[not_bridge], marker='.', alpha=.5)
            plt.scatter(X[:, 0][is_bridge], X[:, 1][is_bridge], c='r', marker='^', alpha=.5)
            plt.quiver(filtered_centers2[:, 0], filtered_centers2[:, 1], filtered_arrows2[:, 0], filtered_arrows2[:, 1], angles='xy', scale_units='xy', scale=1, alpha=.5, headwidth=1)
            plt.axis('off')

            fig3 = plt.figure(figsize=figsize)
            plt.scatter(X[:, 0], X[:, 1], c=pred_labels, marker='.', alpha=.5)
            plt.axis('off')

            fig4 = plt.figure(figsize=figsize)
            plt.scatter(X[:, 0][not_outliers], X[:, 1][not_outliers], c=pred_labels[not_outliers], marker='.', alpha=.5)
            plt.scatter(X[:, 0][outliers], X[:, 1][outliers], c='r', marker='^', alpha=.5)
            plt.quiver(filtered_pred_centers[:, 0], filtered_pred_centers[:, 1], filtered_pred_arrows[:, 0], filtered_pred_arrows[:, 1], angles='xy', scale_units='xy', scale=1, alpha=.5, headwidth=1)
            plt.axis('off')

            fig5 = plt.figure(figsize=figsize)
            plt.scatter(X[:, 0][not_bridge], X[:, 1][not_bridge], c=cluster_labels[not_bridge], marker='.', alpha=.5)
            plt.scatter(X[:, 0][is_bridge], X[:, 1][is_bridge], c='r', marker='^', alpha=.5)
            plt.axis('off')

            fig6 = plt.figure(figsize=figsize)
            plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, marker='.', alpha=.5)
            plt.quiver(all_centers[:, 0], all_centers[:, 1], all_arrows[:, 0], all_arrows[:, 1], angles='xy', scale_units='xy', scale=1, alpha=.5, headwidth=1)
            plt.axis('off')
            # plt.show()

            fig2.savefig(export_path / f'{dt_no_ext}_k{k}_gt_arrows.png', **plot_config)
            fig3.savefig(export_path / f'{dt_no_ext}_k{k}_outlier_clustering.png', **plot_config)
            fig4.savefig(export_path / f'{dt_no_ext}_k{k}_outlier_arrows.png', **plot_config)
            fig5.savefig(export_path / f'{dt_no_ext}_k{k}_bridges.png', **plot_config)
            fig6.savefig(export_path / f'{dt_no_ext}_k{k}_all_arrows.png', **plot_config)

            plt.close('all')
        fig1.savefig(export_path / f'{dt_no_ext}_gt.png', **plot_config)