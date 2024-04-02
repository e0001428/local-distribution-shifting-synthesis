from model import Model
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class RFModel(Model):
    def __init__(self, seed=1234, n_estimator=100, max_depth=5):
        self.seed = seed
        self.model = RandomForestClassifier(n_estimators=n_estimator, max_depth=max_depth, random_state=seed)
        self.name = "Random forest: random state={} n_estimator={} max_depth={}".format(seed,n_estimator,max_depth)