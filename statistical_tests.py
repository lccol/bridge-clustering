import os
import json
import numpy as np
import pandas as pd

from math import sqrt
from typing import Union
from pathlib import Path
from scipy.stats import friedmanchisquare

def friedman_test(df: pd.DataFrame):
    cols = [x for x in df.columns if x != 'dataset']
    data = [df[col] for col in cols]
    res = friedmanchisquare(*data)
    return res

def nemenyi_test(df: pd.DataFrame):
    n = df.shape[0]
    k = df.shape[1]
    cols = [x for x in df.columns if x != 'dataset']

    q_alpha = [1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164]

    selected_q_alpha = q_alpha[k - 2]

    cd = selected_q_alpha * sqrt((k * (k + 1)) / (6 * n))
    ranked_df = df.rank(axis=1, method='average', numeric_only=True, ascending=False)

    mean_ranks = ranked_df[cols].mean()
    result = {f'{cols[x]}_{cols[y]}': {'difference': mean_ranks[x] - mean_ranks[y], 'is_significative': str(abs(mean_ranks[x] - mean_ranks[y]) >= cd)} for x in range(len(cols)) for y in range(x + 1, len(cols))}
    return ranked_df, result, cd, selected_q_alpha

def generate_reports(df: pd.DataFrame, save_basepath: Union[str, Path], save_df: bool=False, save_rankings_df: bool=True, filename: str='report.txt'):
    friedman_out = friedman_test(df)
    nemenyi_out = nemenyi_test(df)

    mask = df.index != 'banana.arff'
    subdf = df[mask]

    if not save_basepath.is_dir():
        save_basepath.mkdir(parents=True)

    ranked_df, differences, cd, q_alpha = nemenyi_out

    if save_df:
        df.to_csv(save_basepath / 'ari_scores.csv')
    if save_rankings_df:
        ranked_df.to_csv(save_basepath / 'rankings.csv')

    with open(save_basepath / filename, 'w') as fp:
        fp.write('ARI scores:')
        fp.write(df.__str__())
        fp.write('\n')

        fp.write('@' * 50)
        fp.write('\n')
        fp.write('Mean ARI:\n')
        fp.write(df.mean().__str__())
        fp.write('\n')

        fp.write('@' * 50)
        fp.write('\n')
        fp.write('Median ARI:\n')
        fp.write(df.median().__str__())
        fp.write('\n')

        fp.write('#' * 50)
        fp.write('\n')
        fp.write('Friedman test:\n')
        fp.write(friedman_out.__str__())
        fp.write('\n')

        fp.write('#' * 50)
        fp.write('\n')
        fp.write('Rankings:\n')
        fp.write(ranked_df.__str__())
        fp.write('\n')
        fp.write('Mean rankings:\n')
        fp.write(ranked_df.mean().__str__())
        fp.write('\n')

        fp.write('_' * 30)
        fp.write('\n')
        fp.write(subdf.__str__())
        fp.write('\n')
        fp.write(f'Mean on shape {subdf.shape}\n')
        fp.write(subdf.mean().__str__())
        fp.write('\n')
        fp.write(f'Median on shape {subdf.shape}\n')
        fp.write(subdf.median().__str__())

        fp.write('@' * 30)
        fp.write('\n')
        fp.write(f'CD: {cd}\n')
        fp.write(f'Q alpha: {q_alpha}\n')
        fp.write('Differences between mean values:\n')
        fp.write(json.dumps(differences, indent=4, sort_keys=True))

if __name__ == '__main__':
    savepath = Path('figures', 'reports_undir_DEF')
    csv_path = savepath / 'ari_scores.csv'
    df = pd.read_csv(csv_path).set_index('dataset').sort_index()
    
    generate_reports(df, savepath, filename='report.txt')