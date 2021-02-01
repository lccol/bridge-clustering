import numpy as np

from .functions import predict_and_cluster

class BridgeClustering:
    def __init__(self, outlier_detection, k: int) -> None:
        self.outlier_detection = outlier_detection
        self.k = k

    def fit(self, X, y=None):
        self.labels_, self.outliers_ = predict_and_cluster(self.outlier_detection, X, self.k)
        return self
    
    def fit_predict(self, X, y=None) -> np.ndarray:
        self.fit(X)
        return self.labels_