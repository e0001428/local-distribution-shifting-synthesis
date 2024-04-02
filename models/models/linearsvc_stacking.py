from model import Model
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import StackingClassifier
import random
import copy

#simple stacking with voting model
class LinearSVCSTACKINGModel(Model):
    def __init__(self, seed=1234, max_iter=1000):
        self.seed = seed
        self.base_model = LinearSVC(max_iter=max_iter, random_state=seed)
        self.name = "linear SVC(Stacking with voting): random state={}".format(seed)
        
    def train(self, X, y):
        n_sample = len(X)
        n_model = max(1,n_sample//20000)
        ne_sample = n_sample//n_model
        self.models = [copy.deepcopy(self.base_model) for i in range(n_model)]
        np.random.seed(12+self.seed)
        indexes = list(range(n_sample))
        random.shuffle(indexes)
        for i in range(n_model):
            X_train = [X[j] for j in indexes[i*ne_sample:min((i+1)*ne_sample,n_sample)]]
            y_train = [y[j] for j in indexes[i*ne_sample:min((i+1)*ne_sample,n_sample)]]
            self.models[i].fit(X_train, y_train)

    def predict(self, X):
        #make prediction
        n_model = len(self.models)
        result = [m.predict(X) for m in self.models]
        res = [[result[j][i] for j in range(n_model)] for i in range(len(X))]
        pdt = [max(set(x), key = x.count) for x in res]
        return pdt