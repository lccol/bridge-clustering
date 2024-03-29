import argparse
import numpy as np
import pandas as pd
import time

from bridge_clustering import BridgeClustering
from pyod.models.lof import LOF
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from pathlib import Path
from collections import defaultdict

def parse():
    parser = argparse.ArgumentParser(description='Script used to launch the scalability tests with synthetic datasets')
    parser.add_argument('-o', type=str, help='Output file name')
    parser.add_argument('-t', type=str, help='Type of scalability test. Must be either "points" or "features".')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse()
    save_filename = args.o
    scal_type = args.t
    assert scal_type in {'points', 'features'}
    
    if scal_type == 'points':
        print('Scalability: points')
        npoints = [100] + list(range(10_000, 150_001, 10_000))
        ndims = [2]
    elif scal_type == 'features':
        print('Scalability: features')
        ndims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600, 700, 800, 900, 1_000]
        npoints = [20_000]

    SEED = 47
    
    savepath = Path('results', 'scalability')
    savepath_full = savepath / save_filename
    
    if not savepath.is_dir():
        savepath.mkdir(parents=True)
        
    df_dict = defaultdict(list)
    
    for npoint in npoints:
        for ndim in ndims:
            print('_' * 50)
            print(f'ndim: {ndim} - npoint: {npoint}')
            
            print('generating dataset...')
            X, y = make_blobs(n_samples=npoint, n_features=ndim, random_state=SEED)
            print(f'generated dataset {X.shape}')
            
            clf = BridgeClustering(outlier_detection=LOF(contamination=.2, n_neighbors=15), k=10)
            
            print('starting fit_predict...')
            start = time.time()
            predict = clf.fit_predict(X)
            delta = time.time() - start
            print(f'end of predict... took {delta}')
            ari = adjusted_rand_score(y, predict)
            
            df_dict['npoints'].append(npoint)
            df_dict['ndim'].append(ndim)
            df_dict['time'].append(delta)
            df_dict['ari'].append(ari)
            
    df = pd.DataFrame(df_dict)
    print(df)
    df.to_csv(savepath_full, index=False)