from model import Model
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

class LinearReg(Model):
    def __init__(self, seed=1234):
        self.seed = seed
        self.model = LinearRegression()
        self.name = "Linear regression"
    
    def predict(self, X):
        prediction = self.model.predict(X)
        return [min(max(x,0),1)for x in prediction]
    
    def test(self, X, y):
        prediction = self.predict(X)
        return [mean_absolute_error(prediction, y), mean_squared_error(prediction, y)]
