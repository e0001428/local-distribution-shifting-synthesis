from model import Model
import numpy as np
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LinearSVRModel(Model):
    def __init__(self, seed=1234, max_iter=1000):
        self.seed = seed
        self.model = LinearSVR(max_iter=max_iter, random_state=seed)
        self.name = "linear SVR: random state={}".format(seed)

    def test(self, X, y):
        prediction = self.model.predict(X)
        return [mean_absolute_error(prediction, y), mean_squared_error(prediction, y)]