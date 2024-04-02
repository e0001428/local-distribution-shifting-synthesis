from model import Model
import numpy as np
from sklearn.svm import SVC

class SVCModel(Model):
    def __init__(self, seed=1234, max_iter=1000, kernel='rbf'):
        self.seed = seed
        self.model = SVC(max_iter=max_iter, kernel=kernel, random_state=seed)
        self.name = "SVC: random state={} kernel={}".format(seed,kernel)
