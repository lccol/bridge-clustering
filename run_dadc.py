import numpy as np
import pandas as pd

from utils import read_arff
from dadc.DADC import DADC

from pathlib import Path
from collections import defaultdict

from sklearn.metrics import adjusted_rand_score

if __name__ == '__main__':
    dataset = Path('datasets')
    savepath = Path('results', 'reports')

    dadc_params = {
        'k': 0.05,
        'cfd_threshold': 0.6
    }

    if not savepath.is_dir():
        savepath.mkdir(parents=True)

    df_dict = defaultdict(list)
    for idx, fullpath in enumerate(dataset.iterdir()):
        if fullpath.stem == 'banana':
            print('skipped banana')
            continue
        print(f'{idx + 1}) clustering dataset {fullpath.stem}')
        X, labels = read_arff(fullpath)

        clf = DADC(**dadc_params)
        pred_labels = clf.runAlgorithm(X)
        print(pred_labels)

        ari = adjusted_rand_score(labels, pred_labels)

        df_dict['dataset'].append(fullpath.stem)
        df_dict['ari'].append(ari)

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