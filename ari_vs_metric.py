import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

if __name__ == '__main__':
    dataset_rw_basepath = Path('real-world')
    basepath = Path('results', 'reports')
    csv_path = basepath / 'ari_vs_recall.csv'
    filter_syn_only = True
    
    rw_datasets = set(x.stem + '.arff' for x in dataset_rw_basepath.iterdir() if x.is_file())

    df = pd.read_csv(csv_path).set_index('dataset').sort_index()
    if filter_syn_only:
        df = df[~df.index.isin(rw_datasets)]

    columns = ['ari', 'recall']

    plot_configuration = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }

    X = df[columns].values
    mask = X[:, 1] >= 0
    
    df = df[columns]
    df = df[mask]

    print(df)
    
    print(df.shape)

    if not filter_syn_only:
        ax = df[~df.index.isin(rw_datasets)].plot.scatter(x='recall', y='ari', c='blue', label='synthetic', grid=True)
        df[df.index.isin(rw_datasets)].plot.scatter(x='recall', y='ari', c='red', label='real world', grid=True, ax=ax)
    else:
        df.plot.scatter(x='recall', y='ari', c='blue', grid=True)
    plt.xlabel('Recall')
    plt.ylabel('Adjusted Rand Score')
    plt.ylim(-0.05, 1.05)
    plt.xlim(-0.05, 1.05)

    # plt.show()
    filename = 'ari_vs_recall_syn_only.png' if filter_syn_only else 'ari_vs_recall_all.png'
    plt.savefig(basepath / filename, **plot_configuration)
    plt.close('all')