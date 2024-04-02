from model import Model
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error

class BRR(Model):
    def __init__(self, seed=1234):
        self.seed = seed
        self.model = BayesianRidge()
        self.name = "BayesianRidge"

    def test(self, X, y):
        prediction = self.model.predict(X)
        return [mean_absolute_error(prediction, y), mean_squared_error(prediction, y)]