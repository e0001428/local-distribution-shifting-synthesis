from abc import ABC, abstractmethod
import numpy as np
from sklearn import metrics

class Model(ABC):
    def __init__(self, seed):
        #define model and model name here
        self.seed = seed
        #self.model = 

    def train(self, x, y):
        #train self.model with data x/y
        np.random.seed(12+self.seed)
        self.model.fit(x,y)

    def test(self, x, y):
        #test self.model with given data x/y
        prediction = self.predict(x)
        return metrics.accuracy_score(prediction, y)

    def predict(self, X):
        #make prediction
        return self.model.predict(X)