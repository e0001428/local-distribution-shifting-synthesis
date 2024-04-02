from model import Model
import numpy as np
from sklearn.svm import LinearSVC

class LinearSVCModel(Model):
    def __init__(self, seed=1234, max_iter=1000):
        self.seed = seed
        self.model = LinearSVC(max_iter=max_iter, random_state=seed)
        self.name = "linear SVC: random state={}".format(seed)
