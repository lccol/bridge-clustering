import numpy as np
import pandas as pd
import sys
import pickle as pkl
import matplotlib.pyplot as plt

from utils import read_arff
from dadc.DADC import DADC

from pathlib import Path
from collections import defaultdict

from sklearn.metrics import adjusted_rand_score

if __name__ == '__main__':
    sys.setrecursionlimit(15000)
    syn_folder = Path('datasets', 'synthetic')
    rw_folder = Path('datasets', 'real-world')
    
    data_list = []
    for fullpath in syn_folder.iterdir():
        data_list.append(fullpath)
    for fullpath in rw_folder.iterdir():
        data_list.append(fullpath)
    
    savepath = Path('results', 'reports')
    image_savepath = savepath / 'images'
    figsize = (9, 9)
    export_images = True
    pred_labels_dict = {}
    
    if not image_savepath.is_dir():
        image_savepath.mkdir(parents=True)

    dadc_params = {
        'k': 0.05,
        'cfd_threshold': 0.6
    }
    
    plot_configuration = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }

    if not savepath.is_dir():
        savepath.mkdir(parents=True)

    df_dict = defaultdict(list)
    for idx, fullpath in enumerate(data_list):
        if fullpath.stem in {'banana', 'cpu', 'iono', 'segment', 'zoo'}:
            print(f'skipped dataset {fullpath.stem}')
            continue
        print(f'{idx + 1}) clustering dataset {fullpath.stem}')
        X, labels = read_arff(fullpath)

        clf = DADC(**dadc_params)
        pred_labels = clf.runAlgorithm(X)
        pred_labels_dict[fullpath.stem] = pred_labels
        print(pred_labels)

        ari = adjusted_rand_score(labels, pred_labels)

        df_dict['dataset'].append(fullpath.stem)
        df_dict['ari'].append(ari)
        
        if export_images:
            fig = plt.figure(figsize=figsize)
            plt.scatter(X[:, 0], X[:, 1], c=pred_labels, marker='.', alpha=.5)
            plt.axis('off')
            fig.savefig(image_savepath / f'{fullpath.stem}_DADC.png', **plot_configuration)
            plt.close(fig)

    df = pd.DataFrame(df_dict).set_index('dataset').sort_index()
    df.to_csv(savepath / 'ari_DADC.csv')

    print(df)

    with open(savepath / 'report_DADC.txt', 'w') as fp:
        fp.write(df.__str__())
        fp.write('\n')
        fp.write(f'Mean values on shape {df.shape}\n')
        fp.write(df.mean().__str__())
        fp.write('\n')
        fp.write(f'Median values on shape {df.shape}\n')
        fp.write(df.median().__str__())
        
    with open(savepath / 'pred_labels.pkl', 'wb') as fp:
        pkl.dump(pred_labels_dict, fp)