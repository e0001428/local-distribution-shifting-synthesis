from model import Model
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class KNNModel(Model):
    def __init__(self, seed=1234, k=5, **kwargs):
        self.seed = seed
        metric = kwargs.get('metric', 'minkowski')
        algorithm = kwargs.get('algorithm', 'auto')
        if metric != 'minkowski':
            algorithm = 'brute'
        self.name = "KNN: k={}".format(k)
        self.model = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm=algorithm)
        if metric != 'minkowski':
            self.name = self.name + ", metric={}".format(metric)