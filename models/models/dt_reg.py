from model import Model
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_squared_error, mean_absolute_error

class DTR(Model):
    def __init__(self, seed=1234, max_depth=5):
        self.seed = seed
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=seed)
        self.name = "Decision Tree: random state={} max_depth={}".format(seed,max_depth)

    def train(self, x, y):
        #train self.model with data x/y
        np.random.seed(12+self.seed)
        self.model.fit(x,y)
        
    def test(self, X, y):
        prediction = self.model.predict(X)
        return [mean_absolute_error(prediction, y), mean_squared_error(prediction, y)]