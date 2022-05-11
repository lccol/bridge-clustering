import numpy as np
import pandas as pd

from denmune import DenMune
from typing import Union, Tuple, Dict, Optional, List

class DenMuneWrapper:
    def __init__(self, k: int, rgn_tsne: bool=False, **kwargs):
        self.k = k
        self.kwargs = kwargs
        self.rgn_tsne = rgn_tsne
        
        self.model = None
        self.labels_ = None
        self.labels = None
        self.validity_ = None
        return
        
    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]]=None):
        assert isinstance(X, (np.ndarray, pd.DataFrame))
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=pd.Int64Index(range(X.shape[1])))
        
        self.model = DenMune(train_data=X,
                             k_nearest=self.k,
                             rgn_tsne=self.rgn_tsne)
        
        self.labels, self.validity_ = self.model.fit_predict(**self.kwargs)
        self.labels_ = self.labels['train']
        return
    
    
    def fit_predict(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[Union[np.ndarray, pd.Series]]=None) -> np.ndarray:
        self.fit(X, y)
        return self.labels_