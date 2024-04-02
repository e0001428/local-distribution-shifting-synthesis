from model import Model
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

class RFR(Model):
    def __init__(self, seed=1234, n_estimator=100, max_depth=5):
        self.seed = seed
        self.model = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth, random_state=seed)
        self.name = "Random forest: random state={} n_estimator={} max_depth={}".format(seed,n_estimator,max_depth)
        
    def test(self, X, y):
        prediction = self.model.predict(X)
        return [mean_absolute_error(prediction, y), mean_squared_error(prediction, y)]