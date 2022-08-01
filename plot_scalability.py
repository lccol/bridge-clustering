import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Union, Dict, List, Set, Optional, Tuple
from functools import reduce

def read_and_merge(file_list: List[Union[str, Path]],
                   join_columns: Optional[List[str]]=None,
                   keep_columns: List[str]=None) -> pd.DataFrame:
    if join_columns is None:
        join_columns = ['npoints', 'ndim']
        
    file_list = [Path(x) for x in file_list]
    df = [pd.read_csv(x).set_index(join_columns).add_suffix('_' + x.stem) for x in file_list]
    merged = reduce(lambda l, r: l.merge(r, left_index=True, right_index=True, how='inner'),\
                        df) \
                .reset_index(drop=False)
    
    if keep_columns is None:
        return merged
    else:
        cols = []
        for x in merged.columns:
            for y in keep_columns:
                if x.startswith(y):
                    cols.append(x)
                    break
        return merged[join_columns + cols]
    
def mean_and_plot(df: pd.DataFrame,
                  col_prefix: str,
                  x_column: str,
                  xlabel: str,
                  ylabel: str,
                  figsize: Tuple[float, float]=None,
                  fontsize: int=20,
                  ticks_fontsize: int=14):
    if figsize is None:
        figsize = (9, 7)
    cols = [x for x in df.columns if x.startswith(col_prefix)]
    df[col_prefix + '_mean'] = df[cols].mean(axis=1)
    
    fig = plt.figure(figsize=figsize)
    plt.plot(df[x_column], df[col_prefix + '_mean'])
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.tick_params(labelsize=ticks_fontsize)
    plt.grid(True)
    
    return df, fig
    
if __name__ == '__main__':
    result_basepath = Path('results', 'scalability')
    image_savepath = result_basepath
    
    dims_list = ['nfeatures1.csv', 'nfeatures2.csv', 'nfeatures3.csv', 'nfeatures4.csv', 'nfeatures5.csv']
    points_list = ['npoints1.csv', 'npoints2.csv', 'npoints3.csv', 'npoints4.csv', 'npoints5.csv']
    
    plot_config = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }
    
    dims_fullpaths = [result_basepath / x for x in dims_list]
    points_list = [result_basepath / x for x in points_list]
    
    merged = read_and_merge(dims_fullpaths, keep_columns=['time'])
    tmp, fig = mean_and_plot(merged,
                             col_prefix='time',
                             x_column='ndim',
                             figsize=(8.5, 6.5),
                             xlabel='Number of features',
                             ylabel='Mean Execution Time (seconds)',
                             ticks_fontsize=14,
                             fontsize=20)
    fig.savefig(image_savepath / 'features.png', **plot_config)
    print(tmp)
    
    merged = read_and_merge(points_list, keep_columns=['time'])
    tmp, fig = mean_and_plot(merged,
                             col_prefix='time',
                             x_column='npoints',
                             figsize=(8.5, 6.5),
                             xlabel='Number of points',
                             ylabel='Mean Execution Time (seconds)',
                             ticks_fontsize=14,
                             fontsize=20)
    fig.savefig(image_savepath / 'points.png', **plot_config)
    print(merged)