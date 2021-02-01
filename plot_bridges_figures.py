import numpy as np
import matplotlib.pyplot as plt

from utils import read_arff
from bridge_clustering import BridgeClustering
from bridge_clustering.functions import determine_bridges, compute_neighbors
from pyod.models.lof import LOF

from pathlib import Path

def plot(X: np.ndarray, bridges: np.ndarray, cluster_labels: np.ndarray):
    # labels must be a boolean array
    not_bridges = ~bridges

    fig = plt.figure()
    plt.scatter(X[:, 0][not_bridges], X[:, 1][not_bridges], c=cluster_labels[not_bridges], marker='.', alpha=.5)
    plt.scatter(X[:, 0][bridges], X[:, 1][bridges], c='red', marker='^', alpha=.5)
    plt.axis('off')
    return fig

if __name__ == '__main__':
    dataset_path = Path('datasets')

    kls = LOF
    args = {'contamination': .2, 'n_neighbors': 15}
    k = 10
    save_path = Path('results', 'bridges_figures')
    final_savepath = save_path / f'{kls.__name__}'

    if not final_savepath.is_dir():
        final_savepath.mkdir(parents=True)

    plot_config = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }

    for dt in dataset_path.iterdir():
        dataset_no_ext = dt.stem

        X, cluster_labels = read_arff(dt)
        distances, indices = compute_neighbors(X, k)
        bridges = determine_bridges(cluster_labels, indices) == 0

        od = kls(**args)
        clf = BridgeClustering(od, k)
        clf.fit(X)

        predicted_labels = clf.labels_
        outliers = clf.outliers_

        bridge_fig = plot(X, bridges, cluster_labels)
        outlier_fig = plot(X, outliers, predicted_labels)

        bridge_savepath = final_savepath / f'{dataset_no_ext}_bridges.png'
        outlier_savepath = final_savepath / f'{dataset_no_ext}_outliers_k{k}.png'

        bridge_fig.savefig(bridge_savepath, **plot_config)
        outlier_fig.savefig(outlier_savepath, **plot_config)

        plt.close('all')