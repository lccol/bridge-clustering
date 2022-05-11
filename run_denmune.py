import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from denmune import DenMune
from utils import read_arff
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from collections import defaultdict

from typing import Union, Dict, Any, Optional, List, Tuple

if __name__ == '__main__':
    dataset_basepath = Path('datasets')
    export_path = Path('results') / 'denmune'
    k_list = list(range(5, 71, 5)) + [6, 39]
#     k_list = [5, 6, 7, 8, 9, 10, 15, 20, 25, 30]
    df_dict = defaultdict(list)
    
    if not export_path.is_dir():
        export_path.mkdir(parents=True)
        
    for k in k_list:
        k_path = export_path / f'k_{k}'
        if not k_path.is_dir():
            k_path.mkdir()
        for p in dataset_basepath.iterdir():
            if not p.is_file():
                continue

            dataset_filename = p.stem
            print('_' * 50)
            print(f'processing dataset {dataset_filename} - k = {k}')

            X, y = read_arff(p)
            X = pd.DataFrame(X, columns=pd.Int64Index([0, 1]))
            y = pd.Series(y, dtype=int)

            model = DenMune(train_data=X, train_truth=y, k_nearest=k, rgn_tsne=False)
            labels, validity = model.fit_predict(show_analyzer=False, show_noise=True)
            
            assert X.shape[0] == y.shape[0] == labels['train'].shape[0]
            
            fig = plt.figure(figsize=(11, 7))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels['train'], marker='.', alpha=.8)
            fig.savefig(k_path / f'{dataset_filename}_{k}.png', bbox_inches='tight')
            plt.close(fig)

            ari = adjusted_rand_score(y, labels['train'])
            nmi = normalized_mutual_info_score(y, labels['train'])

            print(f'ARI: {ari}')
            print(f'NMI: {nmi}')
            
            df_dict['dataset'].append(dataset_filename)
            df_dict['k'].append(k)
            df_dict['ari'].append(ari)
            df_dict['nmi'].append(nmi)
                        
    df = pd.DataFrame(df_dict)
    print(df)
    
    print('Mean scores per k value')
    print(df.groupby('k')[['ari', 'nmi']].mean())
    print('Median scores per k value')
    print(df.groupby('k')[['ari', 'nmi']].median())
    
    df.to_csv(export_path / 'denmune.csv', index=False)