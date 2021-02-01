import os
import sys
import json
import numpy as np
import pandas as pd
import pickle as pkl

from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
from typing import Union, Set, List, Dict, Tuple

from utils import generate_dbscan_config_tree

class Reader(ABC):
    def __init__(self, path: Union[str, Path]) -> None:
        self.path = path
        self.test_datasets = set()

    @abstractmethod
    def analyze(self) -> None:
        pass

    @abstractmethod
    def get_configuration(self, test: str) -> Tuple:
        pass

    @abstractmethod
    def get_all_configurations(self, test: str) -> List[Tuple]:
        pass

    @abstractmethod
    def get_test_ari(self, test: str) -> float:
        pass

    @abstractmethod
    def is_in(self, test: str) -> bool:
        pass

    @abstractmethod
    def to_ari_dataframe(self, remove_extension: bool=True) -> pd.DataFrame:
        pass

    @classmethod
    def _verify_and_insert(cls, dikt: dict, key: str, value) -> dict:
        assert not key in dikt
        dikt[key] = value
        return dikt

class KIndexedList:
    def __init__(self):
        self.data = defaultdict(list)

    def clear(self) -> None:
        for k in self.data:
            self.data[k].clear()

    def insert(self, k: int, index: int) -> None:
        self.data[k].append(index)

    def get(self, k: int) -> List[int]:
        return self.data[k]

    def get_all(self) -> Dict:
        return dict(self.data)

    def gather(self, data_dict: Dict, fixed_level: str) -> List[Dict]:
        result = []
        for k in self.data:
            for idx in self.data[k]:
                subdict = {}
                p, score_dict = data_dict[k][fixed_level][idx]
                subdict['k'] = k
                subdict['ari'] = score_dict['ari']
                subdict['configuration'] = p
                result.append(subdict)

        return result

class ODGridReader(Reader):
    def __init__(self, path: Union[str, Path], model: str) -> None:
        self.path = path
        self.test_datasets = set()
        self.configs = {}
        self.model = model
        self.ari = {}

    def analyze(self):
        with open(self.path, 'rb') as fp:
            results = pkl.load(fp)

        for dataset in results:
            self.test_datasets.add(dataset)
            best_ari = -2
            best_index_dict = KIndexedList()
            for k in results[dataset]:
                for model in results[dataset][k]:
                    if model != self.model:
                        continue
                    for idx, (p, res_dict) in enumerate(results[dataset][k][model]):
                        current_ari = res_dict['ari']
                        if current_ari > best_ari:
                            best_ari = current_ari
                            best_index_dict.clear()
                            best_index_dict.insert(k, idx)
                            self.ari[dataset] = current_ari
                        elif current_ari == best_ari:
                            best_index_dict.insert(k, idx)

            full_list = best_index_dict.gather(results[dataset], self.model)
            res = [(self.model, tmp['configuration'], {'k': tmp['k']}) for tmp in full_list]
            self.configs[dataset] = res
    
    def get_configuration(self, test: str) -> Dict:
        return self.configs[test][0]

    def get_all_configurations(self, test: str) -> List[Dict]:
        return self.configs[test]

    def get_test_ari(self, test: str) -> float:
        return self.ari[test]

    def is_in(self, test: str) -> bool:
        return test in self.test_datasets

    def to_ari_dataframe(self, remove_ext: bool=True) -> pd.DataFrame:
        res = defaultdict(list)
        for dataset in self.test_datasets:
            dt = dataset
            if remove_ext:
                dt = os.path.splitext(dataset)[0]

            res['dataset'].append(dt)
            res['ari'].append(self.get_test_ari(dataset))

        return pd.DataFrame(res)

class ODGridAllReader(ODGridReader):
    def __init__(self, path: Union[str, Path], model: str, dataset_list: Union[Set, List, None]=None) -> None:
        super().__init__(path, model)
        self.configs = defaultdict(list)
        self.dataset_list = dataset_list

    def analyze(self) -> None:
        with open(self.path, 'rb') as fp:
            results = pkl.load(fp)

        for dataset in results:
            if not self.dataset_list is None:
                if not dataset in self.dataset_list:
                    continue
            self.test_datasets.add(dataset)
            for k in results[dataset]:
                for model in results[dataset][k]:
                    if model != self.model:
                        continue
                    for idx, (p, res_dict) in enumerate(results[dataset][k][model]):
                        current_ari = res_dict['ari']
                        res = (self.model, p, {'k': k}, current_ari)
                        self.configs[dataset].append(res)

class ClusteringGridReader(Reader):
    def __init__(self, path: Union[str, Path], model: str) -> None:
        super().__init__(path)
        self.model = model
        self.test_datasets = set()
        self.configs = {}
        self.ari = {}

    def analyze(self) -> None:
        with open(self.path, 'rb') as fp:
            results = pkl.load(fp)

        for dataset in results:
            assert not dataset in self.test_datasets
            self.test_datasets.add(dataset)
            best_ari = -2
            self.configs[dataset] = []
            model = self.model
            for idx, (p, res_dict) in enumerate(results[dataset][model]):
                current_ari = res_dict['ari']
                if current_ari > best_ari:
                    best_ari = current_ari
                    self.configs[dataset].clear()
                    self.configs[dataset].append((self.model, p, {}))
                elif current_ari == best_ari:
                    self.configs[dataset].append((self.model, p, {}))

            self.ari[dataset] = best_ari

    def is_in(self, test: str) -> bool:
        return test in self.test_datasets

    def get_test_ari(self, test: str) -> float:
        return self.ari[test]

    def get_configuration(self, test: str) -> Dict:
        return self.configs[test][0]

    def get_all_configurations(self, test: str) -> List[Dict]:
        return self.configs[test]

    def to_ari_dataframe(self, remove_extension: bool) -> pd.DataFrame:
        res = defaultdict(list)
        for dataset in self.test_datasets:
            dt = dataset
            if remove_extension:
                dt = os.path.splitext(dataset)[0]

            res['dataset'].append(dt)
            res['ari'].append(self.get_test_ari(dataset))

        return pd.DataFrame(res)

class ClusteringGridAllReader(ClusteringGridReader):
    def __init__(self, path: Union[str, Path], model: str, dataset_list: Union[None, List, Set]=None) -> None:
        super().__init__(path, model)
        self.configs = defaultdict(list)
        if not dataset_list is None:
            dataset_list = set(dataset_list)
        self.dataset_list = dataset_list

    def analyze(self) -> None:
        with open(self.path, 'rb') as fp:
            results = pkl.load(fp)

        for dataset in results:
            if not self.dataset_list is None:
                if not dataset in self.dataset_list:
                    continue
            self.test_datasets.add(dataset)
            for p, res_dict in results[dataset][self.model]:
                self.configs[dataset].append((self.model, p, {}, res_dict['ari']))