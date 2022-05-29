import json
import numpy as np
import pandas as pd

from typing import Union, Tuple, Dict
from pathlib import Path
from scipy.io.arff import loadarff
from sklearn import model_selection
from collections import defaultdict

def read_arff(fullpath: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    data, _ = loadarff(fullpath)
    df = pd.DataFrame(data)
    klass = 'class' if 'class' in df.columns else 'CLASS'
    assert klass in df.columns
    df[klass] = pd.factorize(df[klass])[0]
    
#     cols = ['x', 'y'] if {'x', 'y'}.issubset((df.columns)) else ['a0', 'a1']
    cols = [x for x in df.columns if x != klass]
    X = df[cols].values
    y = df[klass].values
    return X, y

def generate_dbscan_config_tree(filepath: Union[str, Path]) -> Dict:
    with open(filepath, 'r') as fp:
        data = json.load(fp)

    result = defaultdict(lambda: {})
    for item in data:
        for config in item['k_values']:
            result[item['dataset']][config['k']] = config['eps']

    return dict(result)

def __compute_over_axis(cm: np.ndarray, axis: int, zero_div: Union[None, float, int]=None) -> np.ndarray:
    diag = np.diag(cm)
    zero_div_mask = cm.sum(axis=axis) == 0
    res = diag / cm.sum(axis=axis).clip(min=1e-4)
    if not zero_div is None:
        res[zero_div_mask] = zero_div
    return  res, zero_div_mask

def __compute_recall(cm: np.ndarray, zero_div: Union[None, float, int]=None) -> np.ndarray:
    return __compute_over_axis(cm, axis=1, zero_div=zero_div)

def compute_recall(cm: np.ndarray, zero_div: Union[None, float, int]=None) -> np.ndarray:
    return __compute_recall(cm, zero_div)[0]

def __compute_precision(cm: np.ndarray, zero_div: Union[None, float, int]=None) -> np.ndarray:
    return __compute_over_axis(cm, axis=0, zero_div=zero_div)

def compute_precision(cm: np.ndarray, zero_div: Union[None, float, int]=None) -> np.ndarray:
    return __compute_precision(cm, zero_div)[0]

def compute_f1(cm: np.ndarray, zero_div: Union[None, float, int]=None) -> np.ndarray:
    precision, prec_mask = __compute_precision(cm, zero_div=zero_div)
    recall, rec_mask = __compute_recall(cm, zero_div=zero_div)
    res = 2 * precision * recall / (precision + recall).clip(min=1e-4)
    if not zero_div is None:
        mask = prec_mask | rec_mask
        res[mask] = zero_div
    return res

def parameter_generator(params):
    pg = model_selection.ParameterGrid(params)
    # print(f'Parameter grid length: {len(pg)}')
    for p in pg:
        yield p