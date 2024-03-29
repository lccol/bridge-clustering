import os
import json
import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Dict, List, Set, Tuple, Union
from pathlib import Path
from readers import ODGridAllReader, ODGridReader, ClusteringGridReader, ClusteringGridAllReader, Reader
from collections import defaultdict

def compute_unique_values(reader: Reader, ignore_fields: Union[List, Set, None]=None, all_configurations: bool=False, none_flag=-10) -> dict:
    unique_values = defaultdict(set)
    for test in reader.test_datasets:
        # print('@' * 30)
        # print(f'dataset {test}')
        
        if not all_configurations:
            best_configuration = [reader.get_configuration(test)]
        else:
            best_configuration = reader.get_all_configurations(test)

        for bc in best_configuration:
            if len(bc) == 3:
                best_kls, best_params, best_k = bc
            else:
                best_kls, best_params, best_k, _ = bc
            for k in best_params:
                if not ignore_fields is None and k in ignore_fields:
                    continue
                unique_values[k].add(best_params[k] if not best_params[k] is None else none_flag)
            
            if isinstance(best_k, dict):
                for k in best_k:
                    if not ignore_fields is None and k in ignore_fields:
                        continue
                    unique_values[k].add(best_k[k])
            else:
                unique_values['k'].add(best_k)
            
    non_numeric_dict = {}
    for k in unique_values:
        bak = unique_values[k]
        unique_values[k] = list(
            sorted(
                filter(lambda x: isinstance(x, (int, float)), unique_values[k])
            )
        )
        non_numeric_dict[k] = list(
               filter(lambda x: not isinstance(x, (int, float)), bak)
        )
        unique_values[k][:0] = non_numeric_dict[k]
    
    unique_values = dict(unique_values)
    return unique_values

def _get_index(dimension_dict: dict, ignore_fields: Union[Set, List, None]=None, none_flag=-10, **parameters) -> np.ndarray:
    coordinates = np.ones(len(dimension_dict), dtype=int) * -1
    for k in parameters:
        if not ignore_fields is None and k in ignore_fields:
            continue
        param_value = parameters[k] if not parameters[k] is None else none_flag
        position_index = -1
        value_index = -1
        for idx in dimension_dict:
            if dimension_dict[idx]['field'] == k:
                assert position_index == -1 and value_index == -1
                position_index = idx
                value_index = dimension_dict[idx]['parameters'].index(param_value)
        assert position_index != -1 and value_index != -1
        coordinates[position_index] = value_index
    return tuple(coordinates)

def compute_dimension_dict(unique_values: dict):
    dimension_dict = {idx: {'field': k, 'parameters': unique_values[k]} for idx, k in enumerate(sorted(unique_values.keys()))}
    return dimension_dict

def _dict_union(a: dict, b: dict):
    res = {}
    for k in a:
        assert not k in res
        res[k] = a[k]

    for k in b:
        assert not k in res
        res[k] = b[k]
    return res

def count_configurations(reader: Reader, unique_values: dict, ignore_fields: Union[Set, List, None]=None, all_configs: bool=False, accumulate: bool=False, none_flag=-10) -> Tuple[np.ndarray, np.ndarray]:
    dims = tuple(len(unique_values[k]) for k in sorted(unique_values.keys()))
    dimension_dict = compute_dimension_dict(unique_values)
    ari_sum = np.zeros(dims)
    is_best_config_sum = np.zeros(dims)
    accumulator = []

    for test in reader.test_datasets:
        if not all_configs:
            params_tuple = [reader.get_configuration(test)]
        else:
            params_tuple = reader.get_all_configurations(test)

        if accumulate:
            ari_sum = np.zeros(dims)
            is_best_config_sum = np.zeros(dims)

        for pt in params_tuple:
            if len(pt) == 3:
                _, params, additional_params = pt
            else:
                _, params, additional_params, ari = pt
            params = _dict_union(params, additional_params)
            coords = _get_index(dimension_dict, ignore_fields, none_flag=none_flag, **params)
            if len(pt) == 3:
                ari_sum[coords] += reader.get_test_ari(test)
            else:
                ari_sum[coords] += ari
            is_best_config_sum[coords] += 1
        if accumulate:
            accumulator.append(ari_sum)
            assert (is_best_config_sum == 1).all()
    if not accumulate:
        return ari_sum, is_best_config_sum, dimension_dict
    else:
        accumulator = np.stack(accumulator, axis=-1)
        return accumulator, dimension_dict

def plot_statistics(dimension_dict: dict,
                    matrix: np.ndarray,
                    selected_features: list,
                    fixed_params: Union[None, Dict]=None,
                    figsize=(11, 9),
                    column_mapper: Union[Dict, None]=None,
                    fontsize: int=8,
                    label_fontsize: int=14,
                    fmt: str='.3g'):
    fixed_params_keys = set(fixed_params.keys()) if not fixed_params is None else set()
    reduction = tuple(idx for idx in dimension_dict if not dimension_dict[idx]['field'] in selected_features and not dimension_dict[idx]['field'] in fixed_params_keys)
    reduction_func = np.sum
    if len(matrix.shape) == len(dimension_dict) + 1:
        reduction = reduction + (-1,)
        reduction_func = np.mean
    to_keep = tuple(k for k in dimension_dict if not k in reduction)
    reduced_matrix = reduction_func(matrix, axis=reduction)
    indexes = [dimension_dict[k]['parameters'] for k in to_keep]
    axis_labels = [dimension_dict[k]['field'] for k in to_keep]

    if not column_mapper is None:
        tmp = [column_mapper[i] for i in axis_labels]
        axis_labels = tmp

    if not fixed_params is None:
        tmp = []
        indexes_2 = []
        axis_labels_2 = []
        for idx in dimension_dict:
            if not dimension_dict[idx]['field'] in selected_features and not dimension_dict[idx]['field'] in fixed_params_keys:
                continue
            if dimension_dict[idx]['field'] in selected_features:
                tmp.append(slice(0, len(dimension_dict[idx]['parameters'])))
                indexes_2.append(dimension_dict[idx]['parameters'])
                axis_labels_2.append(dimension_dict[idx]['field'])
            else:
                tmp.append(dimension_dict[idx]['parameters'].index(fixed_params[dimension_dict[idx]['field']]))
        if not column_mapper is None:
            tmp2 = [column_mapper[i] for i in axis_labels_2]
            axis_labels_2 = tmp2
        
        tmp = tuple(tmp)
        reduced_matrix = reduced_matrix[tmp]
        indexes = indexes_2
        axis_labels = axis_labels_2

    assert len(selected_features) == 1 or len(selected_features) == 2

    if len(selected_features) == 1:
        assert len(indexes) == 1
        reduced_matrix = reduced_matrix[..., np.newaxis]
        df = pd.DataFrame(reduced_matrix, index=indexes[0], columns=['values'])
        # barplot
        fig, ax = plt.subplots(figsize=(figsize))
        df.plot(ax=ax, grid=True, marker='o')
        plt.xlabel(axis_labels[0], fontsize=label_fontsize)
        plt.ylabel('value', fontsize=label_fontsize)
        return fig, reduced_matrix
    else:
        assert len(indexes) == 2
        df = pd.DataFrame(reduced_matrix, columns=indexes[1], index=indexes[0])
        fig, ax = plt.subplots(figsize=(figsize))
        sns.heatmap(df, annot=True, annot_kws={'fontsize': fontsize}, fmt=fmt)
        plt.xlabel(axis_labels[1], fontsize=label_fontsize)
        plt.ylabel(axis_labels[0], fontsize=label_fontsize)
        plt.yticks(rotation=0)
        return fig, reduced_matrix

def get_config_from_index(index: Tuple, dimension_dict: Dict) -> Dict:
    res = {dimension_dict[i]['field']: dimension_dict[i]['parameters'][idx] for i, idx in enumerate(index)}
    return res

def get_config_from_index_array(index_array, dimension_dict: Dict) -> List[Dict]:
    res = [get_config_from_index(idx, dimension_dict) for idx in index_array]
    return res

def find_best_configs(matrix: np.ndarray, reduction_func, threshold: float, dimension_dict: Dict) -> List[Tuple]:
    reduced_matrix = reduction_func(matrix, axis=-1)
    index_array = np.arange(reduced_matrix.size, dtype=np.int).reshape(reduced_matrix.shape)
    mask = reduced_matrix >= threshold

    selected_indexes = index_array[mask].flatten()
    selected_values = reduced_matrix[mask].flatten()
    unraveled_indexes = np.unravel_index(selected_indexes, index_array.shape)
    unraveled_indexes = list(zip(*unraveled_indexes))
    configs = get_config_from_index_array(unraveled_indexes, dimension_dict)
    assert len(configs) == len(selected_values)
    return list(zip(configs, selected_values))

if __name__ == '__main__':
    readpath = Path('other_outlier_det')
    pkl_path = readpath / 'iforest_simpler3.pkl'
    ignore_fields = {'n_jobs', 'random_state'}
    klass = 'IForest'
    savepath = Path('figures_35syn_10rw_DEF', f'{klass}_stats')
    fontsize = 22
    ticks_size = 18
    fontsize_heatmap = 17
    figsize = (8.5, 6.5)
    export_images = False
    
    plt.rcParams.update({'axes.labelsize': fontsize, 'xtick.labelsize': ticks_size, 'ytick.labelsize': ticks_size})

    if not savepath.is_dir():
        savepath.mkdir(parents=True)

    plot_configuration = {
        'bbox_inches': 'tight',
        'pad_inches': 0
    }

    if klass in {'KNN', 'LOF', 'COPOD', 'ECOD', 'COF', 'ABOD'}:
        columns_mapper = {
            'n_neighbors': 'N. of Neighbours',
            'k': 'K',
            'contamination': 'Contamination'
        }
    elif klass == 'HBOS':
        columns_mapper = {
            'n_bins': 'N. of Bins',
            'k': 'K',
            'contamination': 'Contamination',
        }
    elif klass in {'IForest', 'INNE'}:
        columns_mapper = {
            'contamination': 'Contamination',
            'n_estimators': 'N. of Estimators',
            'max_samples': 'Max Samples',
            'bootstrap': 'Bootstrap',
            'behaviour': 'Behavior',
            'k': 'K'
        }
        
    to_keep = set()
    for fullpath in Path('datasets').iterdir():
        if fullpath.is_dir():
            for f in fullpath.iterdir():
                if f.is_file():
                    to_keep.add(f.stem)
    
    assert len(to_keep) == 45


    if klass in {'LOF', 'KNN', 'HBOS', 'COPOD', 'ECOD', 'IForest', 'COF', 'INNE', 'ABOD'}:
        reader = ODGridAllReader(pkl_path, klass, dataset_list=to_keep)
    else:
        reader = ClusteringGridAllReader(pkl_path, klass, dataset_list=to_keep)
    reader.analyze()
    print(f'number of datasets: {len(reader.test_datasets)}')
    

    unique_values = compute_unique_values(reader, ignore_fields, all_configurations=True)
    # ari_sum, is_best_config_sum, dimension_dict = count_configurations(reader, unique_values, ignore_fields, all_configs=True)
    data_matrix, dimension_dict = count_configurations(reader, unique_values, ignore_fields, all_configs=True, accumulate=True)
    
    print(f'dimension dictionary: {dimension_dict}')
    print(f'data matrix shape: {data_matrix.shape}')
    
    mean_scores = data_matrix.mean(axis=-1)
    best_config_coords = np.unravel_index(mean_scores.argmax(), mean_scores.shape)
    best_config = {dimension_dict[idx]['field']: dimension_dict[idx]['parameters'][x] for idx, x in enumerate(best_config_coords)}
    best_mean_ari = mean_scores[best_config_coords]
    
    print(f'Best configuration with ARI {best_mean_ari} for {klass} is:')
    print(best_config)

    if klass == 'HBOS':
        # HBOS
        selected_features = ['n_bins', 'contamination']
        fig, m = plot_statistics(dimension_dict, 
                                 data_matrix, 
                                 selected_features, 
                                 fixed_params={'k': best_config['k']}, 
                                 column_mapper=columns_mapper, 
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize,
                                 fmt='.2g')
        # plt.title('Mean Adjusted Rand Score')
        plt.title('')
        if export_images:
            fig.savefig(savepath / f'{klass}_bins_contamination.png', **plot_configuration)

        selected_features = ['k']
        fig, m = plot_statistics(dimension_dict, 
                                 data_matrix, 
                                 selected_features, 
                                 fixed_params={'contamination': best_config['contamination'], 'n_bins': best_config['n_bins']},
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap, 
                                 label_fontsize=fontsize)
        # plt.title(klass)
        plt.ylabel('value')
        plt.gca().legend().remove()
        plt.ylim([.2, 1.])
        if export_images:
            fig.savefig(savepath / f'{klass}_k.png', **plot_configuration)
    elif klass in {'LOF', 'KNN', 'ABOD', 'COF'}:
        # LOF
        selected_features = ['n_neighbors', 'contamination']
        fig, m = plot_statistics(dimension_dict,
                                 data_matrix,
                                 selected_features,
                                 fixed_params={'k': best_config['k']},
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize)
        # plt.title('Mean Adjusted Rand Score')
        plt.title('')
        if export_images:
            fig.savefig(savepath / f'{klass}_neighbors_contamination.png', **plot_configuration)
    elif klass in {'COPOD', 'ECOD'}:
        selected_features = ['contamination', 'k']
        fig, m = plot_statistics(dimension_dict,
                                 data_matrix,
                                 selected_features,
                                 fixed_params={},
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize,
                                 fmt='.2g')
        plt.title('')
        if export_images:
            fig.savefig(savepath / f'{klass}_k_contamination.png', **plot_configuration)
    elif klass in {'ABOD', 'COF'}:
        selected_features = ['contamination', 'n_neighbors']
        fixed_params = {} if klass == 'COF' else {'method': 'fast'}
        fig, m = plot_statistics(dimension_dict,
                                 data_matrix,
                                 selected_features,
                                 fixed_params=fixed_params,
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize)
        plt.title('')
        if export_images:
            fig.savefig(savepath / f'{klass}_neighbors_contamination.png', **plot_configuration)
            
    elif klass == 'INNE':
        selected_features = ['contamination', 'k']
        fixed_params = {
            'n_estimators': best_config['n_estimators'],
            'max_samples': best_config['max_samples']
        }
        fig, m = plot_statistics(dimension_dict,
                                 data_matrix,
                                 selected_features,
                                 fixed_params=fixed_params,
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize)
        plt.title('')
        if export_images:
            fig.savefig(savepath / f'{klass}_contamination_k.png', **plot_configuration)
            
        selected_features = ['n_estimators', 'max_samples']
        fixed_params = {
            'contamination': best_config['contamination'],
            'k': best_config['k']
        }
        fig, m = plot_statistics(dimension_dict,
                                 data_matrix,
                                 selected_features,
                                 fixed_params=fixed_params,
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize)
        plt.title('')
        if export_images:
            fig.savefig(savepath / f'{klass}_estimators_maxsamples.png', **plot_configuration)

    if klass in {'KNN', 'LOF', 'ABOD', 'COF'}:
    # LOF, KNN, ABOD, COF
        selected_features = ['k']
        fig, m = plot_statistics(dimension_dict,
                                 data_matrix,
                                 selected_features,
                                 fixed_params={'contamination': best_config['contamination'], 'n_neighbors': best_config['n_neighbors']},
                                 column_mapper=columns_mapper,
                                 fontsize=fontsize_heatmap,
                                 label_fontsize=fontsize)
        # plt.title('Mean Adjusted Rand Score')
        plt.title('')
        plt.ylim([.2, 1.])
        plt.gca().legend().remove()
        if export_images:
            fig.savefig(savepath / f'{klass}_k.png', **plot_configuration)

    plt.close('all')
    # plt.show()