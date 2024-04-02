from model import Model
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class KNNRegression(Model):
    def __init__(self, seed=1234, k=5, **kwargs):
        self.seed = seed
        metric = kwargs.get('metric', 'minkowski')
        algorithm = kwargs.get('algorithm', 'auto')
        if metric != 'minkowski':
            algorithm = 'brute'
        self.model = KNeighborsRegressor(n_neighbors=k, metric=metric, algorithm=algorithm)
        self.name = "KNN: k={}".format(k)
        if metric != 'minkowski':
            self.name = self.name + ", metric={}".format(metric)
    
    def test(self, X, y):
        prediction = self.model.predict(X)
        return [mean_absolute_error(prediction, y), mean_squared_error(prediction, y)]
