from diffprivlib.models import RandomForestClassifier as RFDPC
from model import Model
import numpy as np

class RFDPModel(Model):
    def __init__(self, seed=1234, n_estimator=100, max_depth=5, epsilon=500.0):
        self.seed = seed
        self.model = RFDPC(max_depth=max_depth, n_estimator=n_estimator, random_state=seed, epsilon=epsilon)
        self.name = "Random Forest with DP: random state={} n_estimator={} max_depth={} epsilon={}".format(seed,n_estimator,max_depth,epsilon)

    def train(self, X, y):
        #train self.model with data X/y
        np.random.seed(12+self.seed)
        # due to a bug in model that can't take str classes, make a map here
        self.classes_ = np.unique(y)
        self.class2ind_ = {self.classes_[i]:i for i in range(len(self.classes_))}
        self.model.fit(X, [self.class2ind_[yy] for yy in y])
        
    def predict(self, X):
        #make prediction
        pred = self.model.predict(X)
        return [self.classes_[y] for y in pred]