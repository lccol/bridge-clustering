import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import adjusted_rand_score
from bridge_clustering import BridgeClustering
from pyod.models.lof import LOF

from utils import read_arff
from collections import defaultdict

from typing import Union, Tuple, Dict, List

def read_dataset(fullpath: Path) -> Tuple[pd.DataFrame, np.ndarray]:
    if isinstance(fullpath, str):
        fullpath = Path(fullpath)
    if fullpath.suffix == '.csv':
        df = pd.read_csv(fullpath, sep=',')
        X, y = df[['0', '1']], df['label'].to_numpy()
    elif fullpath.suffix == '.arff':
        X, y = read_arff(fullpath)
    elif fullpath.suffix == '.txt':
        df = pd.read_csv(fullpath, delim_whitespace=True, header=None)
        X, y = df[[0, 1]], df[2].to_numpy()
    else:
        raise ValueError(f'invalid path {fullpath}')
    return X, y

if __name__ == '__main__':
    dataset_basepath = Path('datasets_mdpc')
    res_dict = defaultdict(list)
    
    config = {
        'outlier_class': LOF,
        'outlier_params': {
                'contamination': .4,
                'n_neighbors': 12
        },
        'k': 12
    }
    
    mdpc_df = pd.DataFrame([
        ['jain', 1.0],
        ['zelnik5_fourlines', 1.0],
        ['flame', 1.0],
        ['spiral', 1.0],
        ['compound', 0.996804],
        ['R15', 0.985650],
        ['twenty', 1.0],
        ['zelnik1_threecircles', 1.0],
        ['aggregation', 0.807408],
        ['s1', 0.988500]
    ], columns=['dataset', 'mdpc'])
    
    
    for p in dataset_basepath.iterdir():
        if not p.is_file() or p.suffix == '.ipynb' or p.stem == 'birch2':
            continue
            
        X, y = read_dataset(p)
        model = BridgeClustering(config['outlier_class'](**config['outlier_params']), k=config['k'])
        
        y_pred = model.fit_predict(X)
        ari = adjusted_rand_score(y, y_pred)
        
        print(f'{p.stem}: {ari}')
        res_dict['dataset'].append(p.stem)
        res_dict['bac'].append(ari)
        
    res_df = pd.DataFrame(res_dict).merge(mdpc_df, on='dataset')
    res_df.to_csv('mdpc_comparison.csv', index=False)
        
    print(res_df)
    print(res_df.mean())