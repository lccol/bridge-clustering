import pandas as pd
import numpy as np

from collections import defaultdict
from utils import read_arff
from pathlib import Path

if __name__ == '__main__':
    syn_path = Path('datasets', 'synthetic')
    rw_path = Path('datasets', 'real-world')
    
    syn_df_dict = defaultdict(list)
    rw_df_dict = defaultdict(list)
    
    for p in syn_path.iterdir():
        if not p.is_file():
            continue
            
        X, y = read_arff(p)
        syn_df_dict['dataset'].append(p.stem)
        syn_df_dict['npoints'].append(X.shape[0])
        syn_df_dict['nfeatures'].append(X.shape[1])
        syn_df_dict['nclusters'].append(np.unique(y).size)
        
    for p in rw_path.iterdir():
        if not p.is_file() or not p.stem in to_keep:
            continue
            
        X, y = read_arff(p)
        rw_df_dict['dataset'].append(p.stem)
        rw_df_dict['npoints'].append(X.shape[0])
        rw_df_dict['nfeatures'].append(X.shape[1])
        rw_df_dict['nclusters'].append(np.unique(y).size)
        
    syn_df = pd.DataFrame(syn_df_dict) \
                    .sort_values('dataset') \
                    .set_index('dataset')
    rw_df = pd.DataFrame(rw_df_dict) \
                    .sort_values('dataset') \
                    .set_index('dataset')
    
    df = pd.concat([syn_df, rw_df], axis=0) \
            .rename({'npoints': '#points', 'nfeatures': '#features', 'nclusters': '#clusters'}, axis=1)
                
    
    print(df)
    df.to_latex('dataset_stats.tex')