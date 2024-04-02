from model import Model
import numpy as np
from sklearn.linear_model import LogisticRegression

class LRModel(Model):
    def __init__(self, seed=1234, max_iter=1000):
        self.seed = seed
        self.model = LogisticRegression(random_state=seed, max_iter=max_iter)
        self.name = "Logistic regression: random state={}".format(seed)
